# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tune campaign: one baseline -> modeling campaign -> wrap-up.

It starts once the core's required checks hold (the target serves), so nothing is measured
before the baseline (no smoke, no stock reference) and the baseline once (no repeats). It is
background work and never holds the onboarding back: an onboarding that ends while it runs does
not wait for it, and it goes on past the run's close as the tune stage. Runs as the execution
stage's ``tune`` task, pinned to the commit the target is at when it starts (:func:`run_tune_task`),
as ``trtllm-modeling-v2 tune`` (:func:`run_tune_phase`), or, unfinished at the close, from its
checkpoint in a later process (:func:`resume_campaign`). Every phase
checkpoints into ``state.tune`` and is idempotent on resume: a label measured at the same
workload, configs and commit is never re-measured, a turn whose answer is on disk (SOL, roadmap,
candidate, rebase) never re-dispatched. Once ``perf.campaign.max_hours`` (counted from the task's
first start, restarts and retries included) is spent, no new candidate, rebase, member-gated
variant or roadmap is issued; those already issued are measured and judged.
"""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from ..agents.slots import Priority, turn_priority
from ..infra.git import git_checked
from ..operators.coverage import CATALOG_TESTS_DIR, MODELING_V2_PACKAGE
from . import sol as sol_mod
from .attempts import ATTEMPTS_FILE, digest, outcome
from .campaign import Campaign, CandidateState, perf_points, read_decision, revisable
from .harness.microbench import ROUGH_TOLERANCE
from .keep_gate import FAIL, KeepGate, KeepGateError, resolve_keep_gate
from .logits_check import LogitsCheck
from .measure import (
    BenchSpec,
    MeasurementRunner,
    SpoolRunner,
    host_controls,
    load_measurement,
    validate_measurement,
)
from .records import (
    TTFT_TOLERANCE,
    Measurement,
    NoiseFloor,
    PerfLedger,
    compare,
    format_table,
    goal_reached,
)
from .requests import (
    CampaignContext,
    TuneRequests,
    render_rebase_request,
    render_request,
    render_revise_request,
    render_split_request,
    turn_prompt,
)
from .schema import campaign_max_hours, validate_perf_block
from .sol_consensus import METHODS, TOLERANCE, read_rulings, reconcile
from .spec import TuneSpec
from .worktree import open_candidate

PACKAGE = Path(__file__).resolve().parent.parent


def _same_commit(a: str, b: str) -> bool:
    """Two spellings of one commit: one is a prefix (of at least 7) of the other."""
    short = min(len(a), len(b))
    return short >= 7 and a[:short] == b[:short]


def _ttft_tolerance(campaign: Mapping[str, Any]) -> float | None:
    """``perf.campaign.ttft_tolerance``: how much slower a first token may come (null: unjudged)."""
    if "ttft_tolerance" not in campaign:
        return TTFT_TOLERANCE
    value = campaign["ttft_tolerance"]
    return None if value is None else float(value)


#: Where a wrapped-up campaign lists, in the workspace, what it kept that the target does not carry.
UNMERGED_FILENAME = "UNMERGED_OPTIMIZATIONS.md"


def unmerged_optimizations(ledger: Path, candidates: Path, *, branch: str, best_commit: str) -> str:
    """The campaign's kept changes, one section each, for a human to merge; ``""`` with none.

    From the ledger's ``candidate_kept`` records and each candidate's decision file. The delivered
    target does not carry them: they stay on ``branch``.
    """
    kept = []
    for line in ledger.read_text(encoding="utf-8").splitlines() if ledger.is_file() else ():
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if isinstance(entry, dict) and entry.get("kind") == "candidate_kept":
            kept.append(entry)
    if not kept:
        return ""
    lines = [
        "# Performance work left unmerged",
        "",
        f"The tune campaign kept these changes on `{branch}`"
        + (f" (at `{best_commit[:10]}`)" if best_commit else "")
        + ". Each passed the keep rule and the accuracy gate; the delivered target does not "
        "carry them, and merging them is a human's decision.",
        "",
    ]
    for entry in kept:
        slug = str(entry.get("slug") or "?")
        lines += [f"## `{slug}`", ""]
        hypothesis = _decision_field(candidates / slug / "decision.yaml", "hypothesis")
        if hypothesis:
            lines += [hypothesis, ""]
        commits = ", ".join(f"`{str(c)[:10]}`" for c in entry.get("commits") or ()) or "none"
        deltas = ", ".join(
            f"{metric} {float(delta):+.1%}" for metric, delta in (entry.get("deltas") or {}).items()
        )
        gate = entry.get("gate") or {}
        accuracy = "ungated" if entry.get("ungated") else f"{gate.get('score', 'passed')}"
        lines += [
            f"- commits: {commits}",
            f"- measured against its control: {deltas or 'no delta recorded'}",
            f"- accuracy gate: {accuracy}",
            "",
        ]
    return "\n".join(lines)


def _decision_field(path: Path, key: str) -> str:
    """One field of a tuner's decision file, as text; ``""`` when it cannot be read."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return ""
    value = data.get(key) if isinstance(data, dict) else None
    return " ".join(str(value).split()) if value else ""


class TuneStage:
    """One campaign. ``dispatch(role, what, prompt)`` runs one agent turn and returns its text.

    ``attempt`` is the task's attempt (1, then +1 per retry). ``clock`` is the wall clock the
    budget is read on.
    """

    def __init__(
        self,
        spec: TuneSpec,
        tune_state: dict,
        *,
        dispatch: Callable[[str, str, str], str],
        checkpoint: Callable[[], None],
        runner_factory: Callable[[Sequence[str], str], Any] | None = None,
        poll_s: float = 30.0,
        log: Callable[[str], None] = print,
        attempt: int = 1,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.spec = spec
        self.requests = TuneRequests(spec, self._standings)
        self.t = tune_state
        self.dispatch = dispatch
        self.checkpoint = checkpoint
        self.poll_s = poll_s
        self.log = log
        self.clock = clock
        spec.workdir.mkdir(parents=True, exist_ok=True)
        self.data = spec.workdir / "data"
        self.data.mkdir(exist_ok=True)
        self.ledger = PerfLedger(spec.workdir / "perf_ledger.jsonl")
        # Job names carry the workspace: spools outlive a run, and a second run's
        # "tune-t-baseline" must never adopt the first run's receipt. They carry the
        # attempt too (``a2`` ...): a retry's receipts are never the failed attempt's,
        # while a job an earlier attempt left in flight is adopted, not run twice.
        run = hashlib.sha1(str(spec.workdir.resolve()).encode()).hexdigest()[:6]
        self.attempt = max(1, int(attempt))

        def runners(spools: Sequence[str], kind: str) -> Any:
            names = [f"{kind}{run}", *(f"{kind}{run}a{n}" for n in range(2, self.attempt + 1))]
            if runner_factory is not None:
                return runner_factory(spools, names[-1])
            runner = SpoolRunner(spools, prefix=names[-1], earlier=names[:-1])
            if "idle_settle_s" in spec.campaign:
                runner.settle_s = float(spec.campaign["idle_settle_s"])
            return runner

        self.runners = runners
        self.branch_prefix = f"tune/{run}"
        self.perf = self.runners(spec.spools, "tune")
        #: Labels this process gave up on: the session of the job whose run failed last, its
        #: return code, and the job.
        self.failed: dict[str, tuple[str, int, str]] = {}
        #: What a keep must pass, resolved from the contract when the campaign starts.
        self.keep_gate: KeepGate | None = None
        self.t.setdefault("phase", "ground")

    # ------------------------------------------------------------------ helpers
    def _measure(
        self,
        labels: Mapping[str, Sequence[Path]],
        tree: str | None = None,
        *,
        bench: BenchSpec | None = None,
        phase: str = "",
    ) -> dict[str, Measurement | None]:
        return MeasurementRunner(
            bench=self.spec.bench(),
            data=self.data,
            runner=self.perf,
            state=self.t,
            failed=self.failed,
            ledger=self.ledger,
            checkpoint=self.checkpoint,
            log=self.log,
            poll_s=self.poll_s,
        ).measure(labels, tree, bench=bench, phase=phase)

    def _ground_commit(self) -> str:
        """The commit the ground repeats measured: their floor is that tree's alone."""
        baseline = self._baseline()
        commit = ((baseline.meta.get("tree") or {}).get("commit") if baseline else "") or ""
        return str(commit or self.spec.base_commit or "unrecorded")

    def _failure(self, label: str) -> str:
        """Why ``label`` has no measurement: the last run's return code and its job's log."""
        spool, rc, job = self.failed.get(label, ("", 0, label))
        if not spool:
            return "not measured"
        log = Path(spool) / "logs" / f"{self.perf.prefix}-{job}.log"
        return f"rc={rc}, see `{log}`"

    def _answered(self, what: str) -> bool:
        """The turn ``what`` wrote its decision file: it finished, watched or not."""
        return (self.spec.workdir / f"{what}-decision.yaml").exists()

    # ------------------------------------------------------------------- phases
    def _best_tree(self) -> Path:
        """The campaign's ``best`` worktree, created at the spec's base commit the first time."""
        s = self.spec
        return open_candidate(
            s.repo,
            s.workdir / "worktrees",
            "best",
            s.base_commit or "HEAD",
            prefix=self.branch_prefix,
        )

    def ground(self) -> bool:
        """The baseline: one measurement of the onboarded target at the perf configs.

        Onboarding is complete when the campaign starts, so nothing else comes first. The
        baseline must serve the target (``modeling_v2: require``); a keep is judged against
        controls of its own host, the first of them this measurement.
        """
        s = self.spec
        tree = str(self._best_tree())
        baseline = self._measure({"t-baseline": s.identity_configs}, tree=tree).get("t-baseline")
        try:
            if baseline is None:
                raise ValueError(self._failure("t-baseline"))
            validate_measurement(
                baseline,
                s.bench(),
                s.identity_configs,
                git_checked(Path(tree), "rev-parse", "HEAD"),
            )
            served = baseline.meta.get("modeling_v2", "require")
            if served != "require":
                raise ValueError(
                    f"the perf config served `modeling_v2: {served}`: it does not select the "
                    "target (`modeling_v2: require`)"
                )
        except (KeyError, OSError, TypeError, ValueError) as exc:
            self.t["failure"] = f"baseline: {exc}"
            self.log(f"[tune] invalid or missing baseline; campaign cannot start ({exc})")
            return False
        self.t.update(baseline="t-baseline", phase="campaign")
        self.ledger.append("baseline", label="t-baseline")
        self.log(
            "[tune] baseline measured once; keeps are judged on the assumed noise band against "
            "controls of their own host"
        )
        self.checkpoint()
        return True

    def _ground_labels(self) -> list[str]:
        """The measurements of the base tree: the baseline, once."""
        return ["t-baseline"]

    def _context(self, standings: str, attempts: str = "") -> CampaignContext:
        s = self.spec
        return CampaignContext(
            repo=s.repo, package=PACKAGE, campaign_dir=s.workdir, perf_root=s.workdir, target_dir=s.target,
            target_tests=s.target.replace(MODELING_V2_PACKAGE, CATALOG_TESTS_DIR),
            model=s.model, dev_spools=[Path(p) for p in s.dev_spools], env_script=s.env_script, embargo=s.embargo,
            workload=s.describe_workload(),
            standings_table=standings, sister_engines=s.sister_engines, notes=self.requests.sources_note(),
            points=perf_points(s.bench()), attempts=attempts,
        )  # fmt: skip

    def _attempts(self, camp: Campaign) -> str:
        """What the finished candidates found, for a request (``attempts.md`` holds all of it)."""
        return digest(list(camp.state.candidates.values()), self.spec.workdir / ATTEMPTS_FILE)

    def _standings(self) -> str:
        ms = [
            Measurement.load(self.data / label)
            for label in (self.t.get("baseline"),)
            if label and (self.data / label / "meta.json").exists()
        ]
        if not ms:
            return "(no measurement yet)"
        ceilings = self._ceilings()
        sol = "\n\n" + sol_mod.describe(ceilings, ms[0]) if ceilings else ""
        return format_table(ms, ms[0]) + sol

    def _gate_tests(self) -> list[str]:
        """The target's own suite and the modeling_v2 contract tests: run on every tree gated."""
        if not self.spec.campaign.get("gate_tests", True):
            return []
        shared = f"{CATALOG_TESTS_DIR}/"
        return [
            self.spec.target.replace(MODELING_V2_PACKAGE, CATALOG_TESTS_DIR),
            *(
                shared + f"test_modeling_v2_{n}.py"
                for n in ("claims", "no_stale_claims", "target_contract", "routing")
            ),
        ]

    # ------------------------------------------------------------ speed of light
    def _baseline(self) -> Measurement | None:
        label = self.t.get("baseline")
        return load_measurement(self.data, str(label)) if label else None

    def _ceilings(self) -> "sol_mod.SolCeilings | None":
        return sol_mod.from_state(self.t.get("sol") or {})

    def _sol(self) -> None:
        """Ask for the SOL ceilings once, after ground truth: several analyses, one answer.

        Each method of ``perf.sol.methods`` is its own turn, side by side, with one fix turn;
        the analyses that load are reconciled (``sol_consensus``), and what they dispute is
        ruled on once by an adjudication turn that chooses among them. What stays disputed is
        unknown. A turn answered across a restart is not asked again: its output that loads, or
        its decision file, is its answer.
        """
        if not self.spec.sol.get("enabled", True) or "sol" in self.t:
            return
        baseline = self._baseline()
        if baseline is None:
            return
        methods = [str(m) for m in self.spec.sol.get("methods") or METHODS]
        tolerance = float(self.spec.sol.get("tolerance", TOLERANCE))
        with ThreadPoolExecutor(max_workers=len(methods), thread_name_prefix="sol") as pool:
            answers = dict(
                zip(methods, pool.map(lambda m: self._sol_analysis(m, baseline), methods))
            )
        analyses = {m: result for m, (result, _) in answers.items() if result is not None}
        excluded = {m: errors for m, (result, errors) in answers.items() if result is None}
        consensus = reconcile(analyses, baseline.headline(), tolerance=tolerance)
        rulings: dict = {}
        if consensus.disputes:
            rulings = self._sol_rulings(consensus.disputes, list(analyses))
            consensus = reconcile(
                analyses, baseline.headline(), tolerance=tolerance, rulings=rulings
            )
        path = self.spec.workdir / sol_mod.SOL_FILENAME
        ceilings, errors = None, [f"{m}: {e[0]}" for m, e in excluded.items()] or ["no analysis"]
        if analyses:
            path.write_text(yaml.safe_dump(sol_mod.calculate(consensus.inputs), sort_keys=False))
            ceilings, errors = sol_mod.load_sol(path, baseline)
        report = {**consensus.report, "excluded": excluded, "rulings": rulings}
        (self.spec.workdir / "sol").mkdir(parents=True, exist_ok=True)
        (self.spec.workdir / "sol" / "consensus.json").write_text(json.dumps(report, indent=2))
        if ceilings is None:
            self.log(
                f"[tune] no SOL ceilings; the campaign proceeds without them ({errors[0][:240]})"
            )
            self.t["sol"] = {"status": "unavailable", "errors": errors}
            self.ledger.append("sol", status="unavailable", errors=errors)
        else:
            ceilings = replace(ceilings, unknown={**ceilings.unknown, **consensus.unknown})
            self.t["sol"] = sol_mod.as_state(ceilings)
            pct = ceilings.pct_of_sol(baseline)
            status = {k: v.get("status") for k, v in report.items() if k.startswith("headline.")}
            self.ledger.append(
                "sol", status=ceilings.method, pct_of_sol=pct, unknown=dict(ceilings.unknown),
                analyses=list(analyses), excluded=excluded, headline=status, rulings=rulings,
            )  # fmt: skip
            self.log(
                f"[tune] SOL from {len(analyses)} of {len(methods)} analyses ({status}): the "
                f"baseline is at {pct} % of SOL"
            )
        self.checkpoint()

    def _sol_analysis(self, method: str, baseline: Measurement) -> tuple[dict | None, list[str]]:
        """One method's analysis, validated on its own: its turn, and one fix turn."""
        path = self.spec.workdir / "sol" / method / sol_mod.SOL_FILENAME
        result, errors = sol_mod.load_result(path, baseline)
        what = f"sol-{method}"
        if result is None and not self._answered(what):
            request = self.requests.sol_request(baseline, method=method)
            self.dispatch("tuner", what, turn_prompt(request))
            result, errors = sol_mod.load_result(path, baseline)
        if errors and path.exists() and not self._answered(f"{what}-fix"):
            request = self.requests.sol_request(baseline, errors, method=method)
            self.dispatch("tuner", f"{what}-fix", turn_prompt(request))
            result, errors = sol_mod.load_result(path, baseline)
        return result, errors

    def _sol_rulings(self, disputes: list[dict], methods: list[str]) -> dict:
        """The adjudication turn's rulings on ``disputes``; invalid ones are left out, and said."""
        path = self.spec.workdir / "sol" / "rulings.yaml"
        if not path.exists() and not self._answered("sol-adjudicate"):
            request = self.requests.sol_adjudication_request(disputes, methods)
            self.dispatch("tuner", "sol-adjudicate", turn_prompt(request))
        try:
            data = yaml.safe_load(path.read_text()) if path.exists() else None
        except yaml.YAMLError as exc:
            data = None
            self.log(f"[tune] sol: unreadable rulings ({exc})")
        rulings, problems = read_rulings(data, disputes)
        for problem in problems:
            self.log(f"[tune] sol: ruling left out: {problem}")
        return rulings

    def _roadmap(self, n: int = 1, attempts: str = "", ask: bool = True) -> list[dict]:
        """Roadmap ``n``: the first from the baseline profile, later ones from the moved best tree.

        With ``ask=False`` (the wall budget is spent) no turn is asked: it is what is on disk.
        """
        what = "roadmap" if n == 1 else f"roadmap-{n}"
        path = self.spec.workdir / f"{what}.yaml"
        if not ask:
            return (
                [dict(r) for r in yaml.safe_load(path.read_text()) or []] if path.exists() else []
            )
        if not path.exists() and not self._answered(what):
            request = self.requests.roadmap(n, attempts)
            self.dispatch("tuner", what, turn_prompt(request))
        if not path.exists():
            return []
        items = [dict(r) for r in yaml.safe_load(path.read_text()) or []]
        if (
            items
            and self.requests.ledger_mode() != "off"
            and not self.t.get(f"{what}_ledger_checked")
        ):
            errors = self._check_ledger(what, items)
            if errors:
                self.log(
                    f"[tune] {what}: kernel ledger incomplete; the campaign proceeds ({errors[0][:240]})"
                )
                self.ledger.append("kernel_ledger", roadmap=what, errors=errors)
            self.t[f"{what}_ledger_checked"] = True
            self.checkpoint()
        return items

    def _check_ledger(self, what: str, items: Sequence[Mapping]) -> list[str]:
        """The problems perf_optimize's validator finds in the roadmap's kernel ledger."""
        from agent_flow.workflows.perf_optimize.kernel_ledger import (
            QUESTIONS,
            LedgerError,
            cross_validate,
            load_ledger,
        )

        try:
            ledger = load_ledger(self.requests.ledger_path(what))
        except LedgerError as exc:
            return [str(exc)]
        target = float(self.spec.campaign.get("kernel_coverage_pct", 80.0))
        errors = cross_validate(ledger, {"items": [{"id": r.get("slug")} for r in items]}, target)
        if not errors:
            rows = ledger.get("kernels") or []
            answers = {
                q: sum(1 for r in rows if (r.get(q) or {}).get("disposition") == "item")
                for q in QUESTIONS
            }
            self.ledger.append(
                "kernel_ledger",
                roadmap=what,
                rows=len(rows),
                enumerated_share_pct=(ledger.get("coverage") or {}).get("enumerated_share_pct"),
                items_per_question=answers,
            )
            self.log(
                f"[tune] {what}: kernel ledger valid ({len(rows)} kernels, items per question {answers})"
            )
        return errors

    def _resolve_keep_gate(self) -> bool:
        """Resolve the keep gate from the contract; False, the failure recorded, if it cannot."""
        if self.keep_gate is not None:
            return True
        s = self.spec
        try:
            self.keep_gate = resolve_keep_gate(
                s.task, s.campaign, workspace=s.workdir.parent, tests=self._gate_tests()
            )
        except KeepGateError as exc:
            self.t["failure"] = (
                f"accuracy: {exc} (the keep gate is the contract's accuracy gate: fix the "
                "contract or `perf.campaign.accuracy`, or declare `perf.campaign.accuracy: "
                "{gate: off}` to keep without one)"
            )
            self.ledger.append("keep_gate", resolved=False, error=str(exc))
            self.log(f"[tune] the campaign cannot start: {self.t['failure']}")
            self.checkpoint()
            return False
        self.ledger.append("keep_gate", resolved=True, **self.keep_gate.as_record())
        self.log(f"[tune] keep gate: {self.keep_gate.describe()}")
        return True

    def _budget_spent(self) -> bool:
        """The campaign has run ``perf.campaign.max_hours`` or longer, on the spec as it stands.

        The first look that finds it spent is journaled (``budget_spent``); a budget raised since
        is not spent.
        """
        hours = campaign_max_hours(self.spec.campaign)
        started = self.t.get("started_at")
        elapsed = (self.clock() - float(started)) / 3600 if started is not None else 0.0
        if hours is None or elapsed < hours:
            self.t.pop("budget_spent", None)
            return False
        if not self.t.get("budget_spent"):
            self.t["budget_spent"] = {"at": self.clock(), "after_h": round(elapsed, 2)}
            self.ledger.append("budget_spent", max_hours=hours, after_h=round(elapsed, 2))
            self.log(
                f"[tune] the wall budget of {hours:g} h (perf.campaign.max_hours) is spent after "
                f"{elapsed:.1f} h: no new candidate, rebase, variant or roadmap; what is in flight "
                "finishes, then the campaign ends with its best kept tree"
            )
            self.checkpoint()
        return True

    def _ended(self, camp: Campaign, misses: int, max_misses: int, budget: int) -> None:
        """Record why the campaign ended: goal, budget, misses, candidates, roadmaps."""
        if self.t.get("goal_reached"):
            by = "goal"
        elif self.t.get("budget_spent"):
            by = "budget"
        elif misses >= max_misses:
            by = "misses"
        elif len(camp.state.candidates) >= budget:
            by = "iterations"
        else:
            by = "roadmaps"
        started = self.t.get("started_at")
        hours = (self.clock() - float(started)) / 3600 if started is not None else None
        after = round(hours, 2) if hours is not None else None
        self.t["ended"] = {"by": by, "at": self.clock(), "after_h": after}
        self.ledger.append("campaign_end", by=by, after_h=self.t["ended"]["after_h"],
                           best=camp.state.best_commit, kept=list(camp.state.kept))  # fmt: skip
        self.log(
            f"[tune] the campaign ends ({self._end_text() or by}): best "
            f"{camp.state.best_commit[:10]}, kept {list(camp.state.kept)}"
        )

    def _end_text(self) -> str:
        """What ended the campaign, when its wall budget did; ``""`` otherwise."""
        ended = self.t.get("ended") or {}
        if ended.get("by") != "budget":
            return ""
        hours = campaign_max_hours(self.spec.campaign)
        after = (self.t.get("budget_spent") or {}).get("after_h", ended.get("after_h"))
        return (
            f"the wall budget of {hours:g} h (`perf.campaign.max_hours`) ended the campaign"
            if hours is not None
            else "the wall budget ended the campaign"
        ) + (f", spent after {after:.1f} h" if after is not None else "")

    def campaign(self) -> None:
        s, cfg = self.spec, self.spec.campaign
        if self.t.pop("ended", None) is not None:
            # A campaign that ended and is entered again (an explicit `tune`) has a new budget.
            self.t["started_at"] = self.clock()
            self.t.pop("budget_spent", None)
        if not cfg.get("enabled", True):
            self.t["phase"] = "wrap"
            self.checkpoint()
            return
        if not self._resolve_keep_gate():
            return
        camp = Campaign(
            workdir=s.workdir, repo=s.repo, bench=s.bench(), base_configs=s.identity_configs,
            perf_runner=self.runners(s.spools, "tunec"), gate_runner=self.runners(s.dev_spools, "tuneg"),
            keep_gate=self.keep_gate,
            noise=NoiseFloor(),
            noise_commit=self._ground_commit(),
            min_gain=float(cfg.get("min_gain", 0.01)), noise_factor=float(cfg.get("noise_factor", 3.0)),
            ttft_tolerance=_ttft_tolerance(cfg),
            branch_prefix=self.branch_prefix, log=self.log,
            split_variants=bool(cfg.get("split_variants", True)),
            min_projected_gain=float(cfg.get("min_projected_gain", 0.005)),
            micro_noise=float(cfg.get("micro_noise", 0.02)),
            module_tolerance=float(cfg.get("module_tolerance", ROUGH_TOLERANCE)),
            logits=LogitsCheck.from_campaign(cfg),
            checkpoint_keeps=int(cfg.get("checkpoint_keeps", 3)),
            checkpoint_gain=float(cfg.get("checkpoint_gain", 0.05)),
        )  # fmt: skip
        worktrees = s.workdir / "worktrees"
        if not camp.state.best_tree:
            camp.state.best_tree = str(self._best_tree())
            camp.state.best_commit = git_checked(Path(camp.state.best_tree), "rev-parse", "HEAD")
        if not camp.state.control_label:
            # The baseline measured this tree at these configs: its host's control. Other
            # hosts get theirs lazily, behind the first measurement that needs one.
            parts = [p.name for p in s.parts]
            for path in host_controls(self.data, self._ground_labels(), parts):
                camp.set_control(path)
        self._sol()
        self._resume_rebases(camp)
        self._resume_revises(camp)
        # The goal as the spec names it now: a best that already meets it (the baseline, or a
        # resumed campaign's keeps) asks for nothing new; a goal raised since goes on.
        self.t["goal_reached"] = False
        roadmap: list[dict] = []
        if self._goal_met(camp):
            self._reach_goal(camp)
        else:
            n_road = int(self.t.get("roadmaps", 1))
            attempts = self._attempts(camp) if n_road > 1 else ""
            roadmap = [
                r
                for r in self._roadmap(n_road, attempts, ask=not self._budget_spent())
                if r.get("slug") not in camp.state.candidates
            ]
        max_roadmaps = int(cfg.get("max_roadmaps", 2))
        max_par = int(cfg.get("max_parallel_tuners", 4))
        misses, budget = int(self.t.get("misses", 0)), int(cfg.get("max_iterations", 12))
        max_misses = int(cfg.get("max_consecutive_misses", 3))
        futures: dict[str, Future] = {}
        #: Conflicts left without a rebase because the wall budget is spent, each journaled once.
        unrebased: set[str] = set()
        with ThreadPoolExecutor(max_workers=max(1, max_par), thread_name_prefix="tuner") as pool:
            while True:
                spent = self._budget_spent()
                if spent:
                    # Issued already: a turn that answered before a restart is measured, no
                    # other item is asked for.
                    roadmap = [r for r in roadmap if self._answered_candidate(str(r["slug"]))]
                while (
                    roadmap
                    and len(futures) < max_par
                    and len(camp.state.candidates) + len(futures) < budget
                ):
                    item = roadmap.pop(0)
                    slug = str(item["slug"])
                    done = s.workdir / "candidates" / slug / "decision.yaml"
                    if done.exists():
                        # The turn finished while no orchestrator watched it (a restart):
                        # take its decision instead of asking the same question again.
                        try:
                            camp.submit_candidate(slug, done)
                            self.log(f"[tune] {slug}: decision found on resume; not re-dispatched")
                            continue
                        except (OSError, ValueError) as exc:
                            self.log(
                                f"[tune] {slug}: unreadable decision on resume ({exc}); dispatching"
                            )
                    base = camp.state.best_commit or "HEAD"
                    standings, attempts = self._standings(), self._attempts(camp)

                    def start(
                        slug: str = slug,
                        item: Mapping = item,
                        base: str = base,
                        standings: str = standings,
                        attempts: str = attempts,
                    ) -> str:
                        # The worktree checkout (minutes on a parallel filesystem) runs in the
                        # turn's own thread, so turns start side by side.
                        wt = open_candidate(
                            s.repo, worktrees, slug, base, prefix=self.branch_prefix
                        )
                        req = render_request(
                            self._context(standings, attempts), slug=slug, worktree=wt, base=base,
                            branch=f"{self.branch_prefix}/{slug}",
                            hypothesis=str(item.get("hypothesis", "")), job_prefix=slug[:12],
                            deliverables=str(
                                item.get("deliverables", "Commits, entry tests, micro-bench, decision.yaml.")
                            ),
                            modeling_edits=bool(item.get("modeling_edits", True)),
                            share=item.get("share"),
                        )  # fmt: skip
                        return self.dispatch("tuner", slug, turn_prompt(req))

                    futures[slug] = pool.submit(start)
                for slug, fut in list(futures.items()):
                    if fut.done():
                        del futures[slug]
                        decision = s.workdir / "candidates" / slug / "decision.yaml"
                        try:
                            if slug in camp.state.candidates and camp.state.candidates[
                                slug
                            ].status not in ("issued", "conflict", "blocked", "revising"):
                                continue
                            camp.submit_candidate(slug, decision)
                        except (OSError, ValueError) as exc:
                            self.log(f"[tune] {slug}: no valid decision ({exc}); counted as a miss")
                            misses += 1
                            variant = camp.state.candidates.get(slug)
                            if variant is not None and (
                                variant.variant_of
                                and variant.status == "issued"
                                or variant.status == "revising"
                            ):
                                # One variant per split, one revise per candidate: a turn that
                                # answered nothing is dropped.
                                variant.status, variant.reason = (
                                    "dropped",
                                    f"no valid decision ({exc})",
                                )
                                self.t[f"counted-{slug}"] = True
                busy = camp.step()
                if not self.t.get("goal_reached") and self._goal_met(camp):
                    self._reach_goal(camp)
                    roadmap = []
                for st in list(camp.state.candidates.values()):
                    # One rebase per best tree it conflicted with.
                    key = f"rebase-{st.slug}-{st.rebase_ref[:10]}"
                    if (
                        st.status == "conflict"
                        and not self.t.get(key)
                        and st.slug not in futures
                        and st.worktree
                    ):
                        if spent:
                            if st.slug not in unrebased:
                                unrebased.add(st.slug)
                                self.log(
                                    f"[tune] {st.slug}: {st.reason}; no rebase is asked for, "
                                    "the wall budget is spent"
                                )
                            continue
                        # A textual conflict with an earlier keep is no verdict: rebase, retry.
                        self.t[key] = True
                        onto = f"{self.branch_prefix}/best"
                        req = render_rebase_request(
                            self._context(self._standings(), self._attempts(camp)),
                            slug=st.slug,
                            worktree=Path(st.worktree),
                            best_branch=onto,
                            reason=st.reason,
                        )
                        futures[st.slug] = pool.submit(
                            self.dispatch, "tuner", f"rebase-{st.slug}", turn_prompt(req)
                        )
                self._split_turns(camp, pool, futures, budget, spent=spent)
                self._revise_turns(camp, pool, futures, spent=spent)
                for st in camp.state.candidates.values():
                    if st.status in ("kept",) and not self.t.get(f"counted-{st.slug}"):
                        self.t[f"counted-{st.slug}"] = True
                        misses = 0
                    elif st.status in ("reverted", "dropped", "blocked") and not self.t.get(
                        f"counted-{st.slug}"
                    ):
                        self.t[f"counted-{st.slug}"] = True
                        misses += 1
                self.t["misses"] = misses
                self.checkpoint()
                n = int(self.t.get("roadmaps", 1))
                if (
                    not futures
                    and not busy
                    and not roadmap
                    and misses < max_misses
                    and len(camp.state.candidates) < budget
                    and n < max_roadmaps
                    and not self.t.get("goal_reached")
                    and not spent
                ):
                    # The roadmap is spent with budget left: profile the moved best tree.
                    self.t["roadmaps"] = n + 1
                    self.checkpoint()
                    roadmap = [
                        r
                        for r in self._roadmap(n + 1, self._attempts(camp))
                        if r.get("slug") not in camp.state.candidates
                    ]
                    continue
                if (
                    not futures
                    and not busy
                    and (
                        not roadmap or misses >= max_misses or len(camp.state.candidates) >= budget
                    )
                ):
                    break
                time.sleep(self.poll_s)
        # What the campaign kept is scored once, on the tree it ends with.
        final = camp.final_accuracy(lambda: time.sleep(self.poll_s))
        if final is not None:
            self.t["final_accuracy"] = {
                k: final.get(k) for k in ("verdict", "score", "floor", "reason", "commit")
            }
            self.log(f"[tune] final tree's accuracy: {final['verdict']} {final['reason'][:300]}")
        self.t["best_tree"], self.t["best_commit"] = camp.state.best_tree, camp.state.best_commit
        self.t["kept"] = list(camp.state.kept)
        self._ended(camp, misses, max_misses, budget)
        self.t["phase"] = "wrap"
        self.checkpoint()

    def _answered_candidate(self, slug: str) -> bool:
        """``slug``'s tuner turn left a readable decision: it answered, watched or not."""
        try:
            read_decision(self.spec.workdir / "candidates" / slug / "decision.yaml")
        except (OSError, ValueError):
            return False
        return True

    def _split_turns(
        self,
        camp: Campaign,
        pool: ThreadPoolExecutor,
        futures: dict[str, Future],
        budget: int,
        *,
        spent: bool = False,
    ) -> None:
        """Ask a tuner for the member-gated variant of every split candidate that has none.

        Once the candidate or wall budget is spent, or the goal reached, none is asked and the split
        is reverted. A variant a restart cut off is submitted if its decision is on disk, else asked
        again while the wall budget lasts.
        """
        cands = camp.state.candidates
        wall = "the wall budget is spent"
        for st in list(cands.values()):
            if st.status != "split":
                continue
            variant = str(st.split.get("variant") or "")
            if variant and (variant in futures or cands[variant].status != "issued"):
                continue
            if not variant:
                issued = len(cands) + sum(1 for f in futures if f not in cands)
                why = (
                    "the goal is reached" if self.t.get("goal_reached")
                    else f"the budget of {budget} candidates is spent" if issued >= budget
                    else wall if spent
                    else ""
                )  # fmt: skip
                if why:
                    camp.close_split(st.slug, why)
                    continue
                variant = camp.open_variant(st.slug)
            else:
                done = self.spec.workdir / "candidates" / variant / "decision.yaml"
                try:
                    camp.submit_candidate(variant, done)
                    self.log(f"[tune] {variant}: decision found on resume; not re-dispatched")
                    continue
                except (OSError, ValueError):
                    if spent:
                        cands[variant].status, cands[variant].reason = (
                            "dropped",
                            (f"its turn was cut off by a restart and {wall}"),
                        )
                        camp.close_split(st.slug, wall)
                        continue
                    self.log(f"[tune] {variant}: its turn was cut off by a restart; asked again")
            futures[variant] = pool.submit(
                self._ask_variant, st, variant, self._standings(), self._attempts(camp)
            )

    def _ask_variant(self, st: CandidateState, variant: str, standings: str, attempts: str) -> str:
        """One tuner turn: ``st``'s change gated on the mode it lost in, at its measured tree."""
        s, base = self.spec, str(st.split["base"])
        wt = open_candidate(
            s.repo, s.workdir / "worktrees", variant, base, prefix=self.branch_prefix
        )
        req = render_split_request(
            self._context(standings, attempts), slug=st.slug, variant=variant, worktree=wt, base=base,
            branch=f"{self.branch_prefix}/{variant}", best_branch=f"{self.branch_prefix}/best",
            split=st.split, deltas=st.deltas,
            original=s.workdir / "candidates" / st.slug / "decision.yaml",
        )  # fmt: skip
        return self.dispatch("tuner", variant, turn_prompt(req))

    def _revise_turns(
        self, camp: Campaign, pool: ThreadPoolExecutor, futures: dict[str, Future], *, spent: bool
    ) -> None:
        """Ask a candidate's turn once more when it failed for a reason the code can fix.

        Its verdict and numbers go with the request; its miss is counted once the revision is
        measured. None is asked once the wall budget is spent or the goal reached.
        """
        if spent or self.t.get("goal_reached"):
            return
        asked = [
            st
            for st in camp.state.candidates.values()
            if st.slug not in futures and st.worktree and revisable(st)
        ]
        for st in asked:
            was = outcome(st)
            st.revisions += 1
            # Wall time, not the stage clock: a restart compares it to the decision file's mtime.
            self.t[f"revise-{st.slug}"] = {"from": st.status, "at": time.time()}
            st.status = "revising"
            req = render_revise_request(
                self._context(self._standings(), self._attempts(camp)), slug=st.slug,
                worktree=Path(st.worktree), best_branch=f"{self.branch_prefix}/best", reason=st.reason,
            )  # fmt: skip
            camp.ledger.append("candidate_revise", slug=st.slug, outcome=was, reason=st.reason)
            self.log(f"[tune] {st.slug}: {was} ({st.reason[:200]}); its turn revises it once")
            futures[st.slug] = pool.submit(
                self.dispatch, "tuner", f"revise-{st.slug}", turn_prompt(req)
            )
        if asked:
            camp.state.save(camp.state_path)

    def _resume_revises(self, camp: Campaign) -> None:
        """A revise a restart cut off: its decision written since is submitted, else asked again."""
        for st in camp.state.candidates.values():
            asked = self.t.get(f"revise-{st.slug}") or {}
            if st.status != "revising":
                continue
            decision = self.spec.workdir / "candidates" / st.slug / "decision.yaml"
            try:
                if decision.stat().st_mtime > float(asked.get("at", 0)):
                    camp.submit_candidate(st.slug, decision)
                    self.log(f"[tune] {st.slug}: its revise finished during a restart; submitted")
                    continue
            except (OSError, ValueError) as exc:
                self.log(f"[tune] {st.slug}: no valid revise decision ({exc})")
            st.status, st.revisions = str(asked.get("from") or "reverted"), 0
            self.log(f"[tune] {st.slug}: its revise was cut off by a restart; asked again")
        camp.state.save(camp.state_path)

    def _resume_rebases(self, camp: Campaign) -> None:
        """Settle the rebase turns a restart cut off: no future of this process waits on them.

        One whose decision says ``rebased_onto`` the commit asked is submitted; any other is asked
        again.
        """
        for st in camp.state.candidates.values():
            key = f"rebase-{st.slug}-{st.rebase_ref[:10]}"
            if st.status != "conflict" or not self.t.get(key):
                continue
            decision = self.spec.workdir / "candidates" / st.slug / "decision.yaml"
            try:
                data = read_decision(decision)
                if _same_commit(str(data.get("rebased_onto") or ""), st.rebase_ref):
                    camp.submit_candidate(st.slug, decision, data=data)
                    self.log(f"[tune] {st.slug}: its rebase finished during a restart; submitted")
                    continue
            except (OSError, ValueError) as exc:
                self.log(f"[tune] {st.slug}: no valid rebase decision ({exc})")
            self.log(f"[tune] {st.slug}: its rebase was cut off by a restart; asked again")
            del self.t[key]
        self.checkpoint()

    def _best_measurement(self, camp: Campaign) -> Measurement | None:
        """An e2e measurement of this exact best revision, never an earlier local keep."""
        labels = [
            "t-final",
            *(
                c.e2e_label
                for c in reversed(list(camp.state.candidates.values()))
                if c.status == "kept" and c.scope == "e2e"
            ),
            "t-baseline",
        ]
        for label in labels:
            m = load_measurement(self.data, label) if label else None
            if m is not None and all(
                (part.meta.get("tree") or {}).get("commit") == camp.state.best_commit
                for part in (m.parts or {"": m}).values()
            ):
                return m
        return None

    def _goal_ceilings(self) -> "tuple[sol_mod.SolCeilings | None, str]":
        """Only deterministic full-interval bounds can satisfy a SOL goal."""
        ceilings = self._ceilings()
        return ceilings, "public roofline ceilings (explicit assumptions)" if ceilings else ""

    def _goal_text(self) -> str:
        s = self.spec
        goals = []
        if s.goal_ratio is not None:
            goals.append(f"{s.goal_ratio}x the baseline")
        if s.goal_pct_of_sol is not None:
            basis = self._goal_ceilings()[1]
            goals.append(f"{s.goal_pct_of_sol:.0%} of SOL" + (f" on {basis}" if basis else ""))
        return " and ".join(goals)

    def _goal_met(self, camp: Campaign) -> bool:
        """The current best meets every goal the spec names, on every headline metric.

        ``ratio_to_baseline``: that multiple of the baseline; ``pct_of_sol``: that fraction
        of the SOL ceiling :meth:`_goal_ceilings` names.
        """
        s = self.spec
        if s.goal_ratio is None and s.goal_pct_of_sol is None:
            return False
        return self._measurement_meets_goal(self._best_measurement(camp))

    def _measurement_meets_goal(self, best: Measurement | None) -> bool:
        s = self.spec
        if best is None or (s.goal_ratio is None and s.goal_pct_of_sol is None):
            return False
        if s.goal_ratio is not None:
            baseline = self._baseline()
            if baseline is None or not goal_reached(best, baseline, s.goal_ratio):
                return False
        if s.goal_pct_of_sol is not None:
            ceilings, _ = self._goal_ceilings()
            if ceilings is None or not ceilings.reached(best, s.goal_pct_of_sol):
                return False
        return True

    def _reach_goal(self, camp: Campaign) -> None:
        """Record the goal as reached, with the measurement and ceiling basis that decided it."""
        best = self._best_measurement(camp)
        ceilings, basis = self._goal_ceilings()
        pct = ceilings.pct_of_sol(best) if ceilings and best and self.spec.goal_pct_of_sol else {}
        self.t["goal_reached"] = True
        self.ledger.append("goal_reached", goal=self._goal_text(), best=best.label if best else "",
                           basis=basis if self.spec.goal_pct_of_sol else "", pct_of_sol=pct)  # fmt: skip
        self.log(
            f"[tune] goal reached: {self._goal_text()}, by {best.label if best else '?'}"
            + (f" at {pct} % of those ceilings" if pct else "")
            + "; no new candidates"
        )
        self.checkpoint()

    def wrap(self) -> str:
        s = self.spec
        ended = self._end_text()
        labels: dict[str, Sequence[Path]] = {"t-final": list(s.identity_configs)}
        final = None
        failure = ""
        tree = self.t.get("best_tree") or str(self._best_tree())
        commit = git_checked(Path(tree), "rev-parse", "HEAD")
        self.t.update(best_tree=tree, best_commit=commit, goal_reached=False)
        if not self.t.get("kept"):
            # The best tree is the baseline's, measured already: not measured again.
            self.log("[tune] nothing was kept: no final measurement, the baseline stands")
            final_status = "not run: nothing kept, the baseline stands"
        else:
            final = self._measure(labels, tree=tree).get("t-final")
            try:
                if final is None:
                    raise ValueError(self._failure("t-final"))
                validate_measurement(final, s.bench(), s.identity_configs, commit)
            except (KeyError, OSError, TypeError, ValueError) as exc:
                final = None
                failure = f"final e2e unavailable for {commit[:10]}: {exc}"
                final_status = failure
            else:
                baseline = self._baseline()
                if baseline is None:
                    failure = "final e2e has no baseline to validate against"
                    final_status = failure
                else:
                    cmp = compare(
                        final,
                        baseline,
                        NoiseFloor(),
                        min_gain=float(s.campaign.get("min_gain", 0.01)),
                        noise_factor=float(s.campaign.get("noise_factor", 3.0)),
                        ttft_tolerance=_ttft_tolerance(s.campaign),
                    )
                    final_status = f"{cmp.verdict}: {cmp.reason}"
                    if cmp.verdict == "LOSS":
                        failure = "final cumulative e2e regressed: " + cmp.reason
                self.t["goal_reached"] = not failure and self._measurement_meets_goal(final)
        self.t["final_validation"] = {"commit": commit, "status": final_status}
        self.ledger.append("final_validation", **self.t["final_validation"])
        rows = [
            m for label in ("t-baseline",) if (m := load_measurement(self.data, label)) is not None
        ]
        if final is not None:
            rows.append(final)
        summary = s.workdir / "summary.md"
        summary.write_text(
            "# Tune stage summary\n\n"
            + format_table(rows, rows[0] if rows else None)
            + f"\n\nModeling kept (local gains are not e2e claims): {self.t.get('kept', [])}\n"
            + f"\nFinal cumulative validation: {final_status}\n"
            + (
                f"\nFinal tree's accuracy: {accuracy['verdict']}: {accuracy['reason']}\n"
                if (accuracy := self.t.get("final_accuracy"))
                else ""
            )
            + (
                f"\nGoal: {self._goal_text()}, "
                f"{'reached' if self.t.get('goal_reached') else 'not reached'}\n"
                if s.goal_ratio or s.goal_pct_of_sol
                else ""
            )
            + (
                "\n## Speed of light\n\n" + sol_mod.render_table(self._ceilings(), rows) + "\n"
                if self._ceilings() and rows
                else ""
            )
            + f"\nBest tree: {self.t.get('best_tree', '')} @ {str(self.t.get('best_commit', ''))[:10]}\n"
            + (f"\nEnded: {ended}.\n" if ended else "")
        )
        unmerged = unmerged_optimizations(
            s.workdir / "perf_ledger.jsonl",
            s.workdir / "candidates",
            branch=f"{self.branch_prefix}/best",
            best_commit=str(self.t.get("best_commit") or ""),
        )
        if unmerged:
            path = s.workdir.parent / UNMERGED_FILENAME
            path.write_text(unmerged, encoding="utf-8")
            self.log(f"[tune] what the delivered target does not carry: {path}")
        if failure:
            self.t["failure"] = failure
            self.checkpoint()
            return "tune-failed:final"
        if (self.t.get("final_accuracy") or {}).get("verdict") == FAIL:
            self.t["failure"] = "the final tree fails the accuracy gate: " + str(
                self.t["final_accuracy"].get("reason")
            )
            self.checkpoint()
            return "tune-failed:accuracy"
        self.t["phase"] = "done"
        self.checkpoint()
        return "tuned"

    def run(self) -> str | None:
        self.t.pop("failure", None)
        # The wall budget counts from here, across restarts and retries.
        self.t.setdefault("started_at", self.clock())
        if self.attempt > 1:
            self.log(f"[tune] attempt {self.attempt}: jobs queue as `{self.perf.prefix}-<label>`")
        # Before anything is measured: no keep gate (and no `gate: off`), no campaign.
        if (
            self.t["phase"] in ("ground", "campaign")
            and self.spec.campaign.get("enabled", True)
            and not self._resolve_keep_gate()
        ):
            return "tune-failed:accuracy"
        if self.t["phase"] == "ground" and not self.ground():
            return "tune-failed:baseline"
        if self.t["phase"] == "campaign":
            self.campaign()
        if self.t["phase"] == "wrap":
            return self.wrap()
        return "tuned" if self.t["phase"] == "done" else None


def _stage_for(
    wf: Any,
    state: Any,
    base_commit: str = "",
    attempt: int = 1,
) -> "TuneStage | str":
    """The stage the task's perf block describes, or the verdict that it cannot start."""
    errors = validate_perf_block(wf.task_data)
    if errors or not wf.task_data.get("perf"):
        wf.journal.event("tune not started", "; ".join(errors) or "task.yaml has no perf block")
        return "tune-failed:spec"
    target = getattr(getattr(wf, "_preflight", None), "target_rel", "") or ""
    spec = TuneSpec.from_task(wf.task_data, wf.workspace, target=str(target))
    spec.base_commit = base_commit

    def dispatch(role: str, what: str, prompt: str) -> str:
        # A campaign's turns use only the agent slots the feature work leaves idle.
        with turn_priority(Priority.BACKGROUND):
            return wf._run_turn(role, what, prompt, unit="tune")

    return TuneStage(spec, state.tune, dispatch=dispatch, checkpoint=lambda: wf._checkpoint(state),
                     log=lambda m: wf.journal.event("tune", m), attempt=attempt)  # fmt: skip


def run_tune_phase(wf: Any, state: Any, log: Any) -> str | None:
    """``trtllm-modeling-v2 tune``: tune the target the task names, as the checkout has it."""
    stage = _stage_for(wf, state)
    return stage if isinstance(stage, str) else stage.run()


def run_tune_task(
    wf: Any,
    state: Any,
    base_commit: str,
    attempt: int = 1,
) -> tuple[str, str]:
    """The execution stage's ``tune`` task: the campaign on ``base_commit``.

    Returns the task's decision (``TUNED`` when the campaign wrapped up) and one line for the
    planner: the best tree and what was kept, or what failed. A resumed run keeps its first pin.
    """
    base = str(state.tune.setdefault("base_commit", base_commit))
    # A campaign the onboarding closes beside resumes as this attempt (:func:`resume_campaign`).
    state.tune["attempt"] = attempt
    stage = _stage_for(wf, state, base, attempt)
    verdict = stage if isinstance(stage, str) else stage.run()
    if verdict != "tuned":
        return f"TUNE_FAILED ({verdict or 'incomplete'})", str(state.tune.get("failure") or "")
    t = state.tune
    ended = stage._end_text()
    line = (
        f"tuned from {base[:10]}: best branch `{stage.branch_prefix}/best` at "
        f"{str(t.get('best_commit', ''))[:10]}, kept {t.get('kept', [])}, goal "
        f"{'reached' if t.get('goal_reached') else 'not reached'}"
        + (f"; {ended}" if ended else "")
        + f"; summary `{stage.spec.workdir / 'summary.md'}`"
    )
    return "TUNED", line


def resume_campaign(wf: Any, state: Any) -> str | None:
    """A campaign the onboarding closed beside, from its checkpoint; the tune stage's verdict.

    On its first pin, as its last attempt, whose queued jobs it adopts. ``None``: unfinished.
    """
    stage = _stage_for(
        wf, state, str(state.tune.get("base_commit") or ""), int(state.tune.get("attempt") or 1)
    )
    return stage if isinstance(stage, str) else stage.run()


def campaign_verdict(decision: str) -> str | None:
    """The tune stage's verdict from a ``tune`` task's decision; ``None`` when it did not end."""
    first = decision.partition("\n")[0].strip()
    if first == "TUNED":
        return "tuned"
    failed = first.removeprefix("TUNE_FAILED (").removesuffix(")")
    return failed if failed != first and failed.startswith("tune-failed") else None
