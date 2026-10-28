# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tune campaign: perf smoke -> ground truth -> modeling campaign -> wrap-up.

Runs as the execution stage's ``tune`` task, pinned to the commit the target is at when it starts
(:func:`run_tune_task`), or as ``trtllm-modeling-v2 tune`` (:func:`run_tune_phase`). Every phase
checkpoints into ``state.tune`` and is idempotent on resume: a label measured at the same
workload, configs and commit is never re-measured, a turn whose answer is on disk (SOL, roadmap,
candidate, rebase) never re-dispatched. Once ``perf.campaign.max_hours`` (counted from the task's
first start, restarts and retries included) is spent, no new candidate, rebase, member-gated
variant or roadmap is issued; those already issued are measured and judged.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import yaml

from ..agents.slots import Priority, turn_priority
from ..infra.git import git_checked
from . import sol as sol_mod
from .campaign import Campaign, CandidateState, read_decision
from .harness.bench import resolve_config
from .keep_gate import KeepGate, KeepGateError, resolve_keep_gate
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
    ground_floors,
)
from .requests import (
    CampaignContext,
    TuneRequests,
    render_rebase_request,
    render_request,
    render_split_request,
    turn_prompt,
)
from .schema import campaign_max_hours, validate_perf_block
from .serving_config import write_serving_config
from .spec import TuneSpec
from .worktree import open_candidate

PACKAGE = Path(__file__).resolve().parent.parent
#: The perf smoke before the baseline: con 1, this many prompts of this many output tokens,
#: one job per world size, each stopped at this wall (``perf.campaign.smoke_wall``).
SMOKE_PROMPTS = 4
SMOKE_OSL = 32
SMOKE_WALL = "45m"


def _world_size(configs: Sequence[Path]) -> int:
    """The ranks one server of these configs runs, as ``bench.py`` launches it."""
    config = resolve_config(list(configs), [])
    world = 1
    for key in ("tensor_parallel_size", "pipeline_parallel_size", "context_parallel_size"):
        world *= int(config.get(key) or 1)
    return world


def _floor_line(same: NoiseFloor, cross: NoiseFloor) -> str:
    """The ground floors in one journal line: the same-host floor, and the spread across hosts."""

    def widest(floor: NoiseFloor) -> str:
        metric = max(floor.spread, key=floor.spread.get)
        return f"{floor.spread[metric]:.2%} ({metric})"

    line = (
        f"noise floor: same-host spread up to {widest(same)} over {same.samples} runs"
        if same.spread
        else "noise floor: not measured; using the conservative assumed band (no automatic repeats)"
    )
    if cross.spread:
        line += (
            f"; across hosts up to {widest(cross)}, not a floor: a keep is judged on its own host"
        )
    return line


def _same_commit(a: str, b: str) -> bool:
    """Two spellings of one commit: one is a prefix (of at least 7) of the other."""
    short = min(len(a), len(b))
    return short >= 7 and a[:short] == b[:short]


@contextlib.contextmanager
def _tuner_pool(workers: int, stopped: Callable[[], bool]) -> Iterator[ThreadPoolExecutor]:
    """The pool tuner turns run on; on leaving it, a stopped campaign does not wait for them."""
    pool = ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="tuner")
    try:
        yield pool
    finally:
        stop = stopped()
        pool.shutdown(wait=not stop, cancel_futures=stop)


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
    budget is read on. ``should_stop`` says the run no longer needs the campaign (full support
    holds): it then wraps up as when its wall budget is spent, and drops what is not yet measuring.
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
        should_stop: Callable[[], bool] = lambda: False,
    ) -> None:
        self.spec = spec
        self.requests = TuneRequests(spec, self._standings)
        self.should_stop = should_stop
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

    def _smoke_configs(self) -> dict[int, tuple[str, list[Path]]]:
        """The first perf config of each world size served: ``{world: (name, configs)}``."""
        s = self.spec
        runs = [(p.name, list(p.configs)) for p in s.parts] or [("identity", s.identity_configs)]
        out: dict[int, tuple[str, list[Path]]] = {}
        for name, configs in runs:
            out.setdefault(_world_size(configs), (name, list(configs)))
        return dict(sorted(out.items()))

    def smoke(self) -> bool:
        """The perf preflight, before the baseline: one short serve per world size.

        Con 1, ``SMOKE_PROMPTS`` prompts of ``SMOKE_OSL`` tokens, on the perf sessions and the
        tree the baseline measures. No automatic reruns. Skipped
        once the baseline is measured, and with ``perf.campaign.smoke: false``.
        """
        s = self.spec
        if not s.campaign.get("smoke", True) or (self.data / "t-baseline" / "meta.json").exists():
            return True
        try:
            runs = self._smoke_configs()
        except (OSError, yaml.YAMLError) as exc:
            self.t["failure"] = f"smoke: the perf configs do not read ({exc})"
            self.log(f"[tune] the perf smoke cannot start: {self.t['failure']}")
            self.checkpoint()
            return False
        labels = {f"t-smoke-w{world}": configs for world, (_, configs) in runs.items()}
        bench = replace(
            s.bench(),
            isl=min([p.isl for p in s.parts] or [s.isl]),
            osl=SMOKE_OSL,
            concurrencies=[1],
            parts=(),
            wall=str(s.campaign.get("smoke_wall", SMOKE_WALL)),
            extra_args=(
                "--num-prompts", f"1:{SMOKE_PROMPTS}", "--warmup-concurrency", "1",
                "--warmup-prompts", "1", "--warmup-osl", "8",
            ),
        )  # fmt: skip
        tree = str(self._best_tree()) if s.base_commit else None
        got = self._measure(labels, tree=tree, bench=bench, phase="smoke")
        problems = []
        for world, (name, _) in runs.items():
            label = f"t-smoke-w{world}"
            m = got.get(label)
            if m is None:
                problems.append(f"world size {world} (`{name}`): {self._failure(label)}")
            elif (served := m.meta.get("modeling_v2", "require")) != "require":
                problems.append(
                    f"world size {world} (`{name}`) served `modeling_v2: {served}`: the perf "
                    "config does not select the target (`modeling_v2: require`)"
                )
        self.ledger.append("smoke", passed=not problems, worlds=sorted(runs), problems=problems)
        if problems:
            self.t["failure"] = "smoke: " + "; ".join(problems)
            self.log(
                "[tune] the perf smoke failed; the baseline is not started: " + "; ".join(problems)
            )
        else:
            self.log(f"[tune] the perf smoke passed at world sizes {sorted(runs)}")
        self.checkpoint()
        return not problems

    def ground(self) -> bool:
        """The baseline, its noise repeats and the stock reference, over every perf node at once.

        A keep is judged against controls of its own host, so its floor is a same-host
        spread (:func:`records.ground_floors`); the spread across hosts judges nothing.
        """
        s = self.spec
        stock = write_serving_config(s.workdir / "configs" / "stock.yaml", {"modeling_v2": "off"})
        ground = self._ground_labels()
        labels: dict[str, Sequence[Path]] = {label: s.identity_configs for label in ground}
        if s.reference == "stock":
            labels["t-reference"] = [*s.identity_configs, stock]
        tree = str(self._best_tree())
        got = self._measure(labels, tree=tree)
        baseline = got.get("t-baseline")
        try:
            if baseline is None:
                raise ValueError(self._failure("t-baseline"))
            validate_measurement(
                baseline,
                s.bench(),
                s.identity_configs,
                git_checked(Path(tree), "rev-parse", "HEAD"),
            )
        except (KeyError, OSError, TypeError, ValueError) as exc:
            self.t["failure"] = f"baseline: {exc}"
            self.log(f"[tune] invalid or missing baseline; campaign cannot start ({exc})")
            return False
        # The campaign's first floor, of the baseline tree; every later best tree is judged on
        # a floor of its own (``Campaign._floor``).
        same, cross = ground_floors([got[label] for label in ground if got.get(label)])
        self.t.update(baseline="t-baseline", reference="t-reference" if s.reference == "stock" else "",
                      noise_spread=same.spread, noise_samples=same.samples,
                      cross_host_spread=cross.spread, phase="campaign")  # fmt: skip
        self.ledger.append("noise_floor", phase="ground", spread=same.spread, samples=same.samples,
                           cross_host=cross.spread)  # fmt: skip
        self.log("[tune] " + _floor_line(same, cross))
        self.checkpoint()
        return True

    def _ground_labels(self) -> list[str]:
        """The baseline and its noise repeats: one measurement of the base tree, taken again."""
        return ["t-baseline", *(f"t-baseline-r{i}" for i in range(1, self.spec.noise_repeats + 1))]

    def _context(self, standings: str) -> CampaignContext:
        s = self.spec
        return CampaignContext(
            repo=s.repo, package=PACKAGE, campaign_dir=s.workdir, perf_root=s.workdir, target_dir=s.target,
            target_tests=s.target.replace("tensorrt_llm/_torch/_experimental/", "tests/unittest/_torch/"),
            model=s.model, dev_spools=[Path(p) for p in s.dev_spools], env_script=s.env_script, embargo=s.embargo,
            workload=s.describe_workload(),
            standings_table=standings, sister_engines=s.sister_engines, notes=self.requests.sources_note(),
        )  # fmt: skip

    def _standings(self) -> str:
        ms = [
            Measurement.load(self.data / label)
            for label in (self.t.get("baseline"), self.t.get("reference"))
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
        shared = "tests/unittest/_torch/modeling_v2/"
        return [
            self.spec.target.replace(
                "tensorrt_llm/_torch/_experimental/", "tests/unittest/_torch/"
            ),
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
        """Ask for the SOL ceilings once, after ground truth; one fix turn, then go on without.

        A turn answered across a restart is not asked again: a ``sol.yaml`` that loads, or the
        turn's decision file, is its answer.
        """
        if not self.spec.sol.get("enabled", True) or "sol" in self.t:
            return
        baseline = self._baseline()
        if baseline is None:
            return
        path = self.spec.workdir / sol_mod.SOL_FILENAME
        ceilings, errors = sol_mod.load_sol(path, baseline)
        if ceilings is None and not self._answered("sol"):
            self.dispatch("tuner", "sol", turn_prompt(self.requests.sol_request(baseline)))
            ceilings, errors = sol_mod.load_sol(path, baseline)
        if errors and path.exists() and not self._answered("sol-fix"):
            self.dispatch(
                "tuner", "sol-fix", turn_prompt(self.requests.sol_request(baseline, errors))
            )
            ceilings, errors = sol_mod.load_sol(path, baseline)
        if ceilings is None:
            self.log(
                f"[tune] no SOL ceilings; the campaign proceeds without them ({errors[0][:240]})"
            )
            self.t["sol"] = {"status": "unavailable", "errors": errors}
            self.ledger.append("sol", status="unavailable", errors=errors)
        else:
            self.t["sol"] = sol_mod.as_state(ceilings)
            pct = ceilings.pct_of_sol(baseline)
            self.ledger.append(
                "sol", status=ceilings.method, pct_of_sol=pct, unknown=dict(ceilings.unknown)
            )
            self.log(f"[tune] SOL ({ceilings.method}): the baseline is at {pct} % of SOL")
        self.checkpoint()

    def _roadmap(self, n: int = 1, tried: Sequence[str] = (), ask: bool = True) -> list[dict]:
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
            request = self.requests.roadmap(n, tried)
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

    def _stop_requested(self) -> bool:
        """The run asked the campaign to wrap up (``should_stop``); journaled the first time."""
        if not self.should_stop():
            return False
        if not self.t.get("stopped"):
            self.t["stopped"] = {"at": self.clock()}
            self.ledger.append("stopped", reason="support-complete")
            self.log(
                "[tune] full support holds on the target: no new candidate, rebase, variant or "
                "roadmap; candidates not yet measured are dropped, what is measuring finishes, "
                "then the campaign ends with its best kept tree"
            )
            self.checkpoint()
        return True

    def _ended(self, camp: Campaign, misses: int, max_misses: int, budget: int) -> None:
        """Record why the campaign ended: goal, support, budget, misses, candidates, roadmaps."""
        if self.t.get("goal_reached"):
            by = "goal"
        elif self.t.get("stopped"):
            by = "support-complete"
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
        """What ended the campaign, when its wall budget or full support did; ``""`` otherwise."""
        ended = self.t.get("ended") or {}
        if ended.get("by") == "support-complete":
            return "full support held on the target, so the campaign wrapped up"
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
            self.t.pop("stopped", None)
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
            noise=NoiseFloor(spread=dict(self.t.get("noise_spread") or {}),
                             samples=int(self.t.get("noise_samples", 0))),
            noise_commit=self._ground_commit(),
            min_gain=float(cfg.get("min_gain", 0.01)), noise_factor=float(cfg.get("noise_factor", 3.0)),
            ttft_tolerance=_ttft_tolerance(cfg),
            branch_prefix=self.branch_prefix, log=self.log,
            split_variants=bool(cfg.get("split_variants", True)),
        )  # fmt: skip
        worktrees = s.workdir / "worktrees"
        if not camp.state.best_tree:
            camp.state.best_tree = str(self._best_tree())
            camp.state.best_commit = git_checked(Path(camp.state.best_tree), "rev-parse", "HEAD")
        if not camp.state.control_label:
            # The ground baseline and its repeats measured this tree at these configs: each
            # time a node ran every member is one of its host's controls. Other hosts get
            # theirs lazily, behind the first measurement that needs one.
            parts = [p.name for p in s.parts]
            for path in host_controls(self.data, self._ground_labels(), parts):
                camp.set_control(path)
        self._sol()
        self._resume_rebases(camp)
        # The goal as the spec names it now: a best that already meets it (the baseline, or a
        # resumed campaign's keeps) asks for nothing new; a goal raised since goes on.
        self.t["goal_reached"] = False
        roadmap: list[dict] = []
        if self._goal_met(camp):
            self._reach_goal(camp)
        else:
            n_road = int(self.t.get("roadmaps", 1))
            tried = list(camp.state.candidates) if n_road > 1 else ()
            roadmap = [
                r
                for r in self._roadmap(n_road, tried=tried, ask=not self._budget_spent())
                if r.get("slug") not in camp.state.candidates
            ]
        max_roadmaps = int(cfg.get("max_roadmaps", 2))
        max_par = int(cfg.get("max_parallel_tuners", 4))
        misses, budget = int(self.t.get("misses", 0)), int(cfg.get("max_iterations", 12))
        max_misses = int(cfg.get("max_consecutive_misses", 3))
        futures: dict[str, Future] = {}
        #: Conflicts left without a rebase because the wall budget is spent, each journaled once.
        unrebased: set[str] = set()
        with _tuner_pool(max_par, lambda: bool(self.t.get("stopped"))) as pool:
            while True:
                stopping = self._stop_requested()
                spent = self._budget_spent() or stopping
                if stopping:
                    # The run is fully supported: nothing not yet measuring is measured.
                    roadmap = []
                elif spent:
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
                    standings = self._standings()

                    def start(
                        slug: str = slug,
                        item: Mapping = item,
                        base: str = base,
                        standings: str = standings,
                    ) -> str:
                        # The worktree checkout (minutes on a parallel filesystem) runs in the
                        # turn's own thread, so turns start side by side.
                        wt = open_candidate(
                            s.repo, worktrees, slug, base, prefix=self.branch_prefix
                        )
                        req = render_request(
                            self._context(standings), slug=slug, worktree=wt, base=base,
                            branch=f"{self.branch_prefix}/{slug}",
                            hypothesis=str(item.get("hypothesis", "")), job_prefix=slug[:12],
                            deliverables=str(
                                item.get("deliverables", "Commits, entry tests, micro-bench, decision.yaml.")
                            ),
                            modeling_edits=bool(item.get("modeling_edits", True)),
                        )  # fmt: skip
                        return self.dispatch("tuner", slug, turn_prompt(req))

                    futures[slug] = pool.submit(start)
                for slug, fut in list(futures.items()):
                    if fut.done() and stopping:
                        del futures[slug]
                        self.log(
                            f"[tune] {slug}: answered after the run was fully supported; dropped"
                        )
                        continue
                    if fut.done():
                        del futures[slug]
                        decision = s.workdir / "candidates" / slug / "decision.yaml"
                        try:
                            if slug in camp.state.candidates and camp.state.candidates[
                                slug
                            ].status not in ("issued", "conflict", "blocked"):
                                continue
                            camp.submit_candidate(slug, decision)
                        except (OSError, ValueError) as exc:
                            self.log(f"[tune] {slug}: no valid decision ({exc}); counted as a miss")
                            misses += 1
                            variant = camp.state.candidates.get(slug)
                            if (
                                variant is not None
                                and variant.variant_of
                                and variant.status == "issued"
                            ):
                                # One variant per split: a turn that answered nothing is dropped.
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
                            self._context(self._standings()),
                            slug=st.slug,
                            worktree=Path(st.worktree),
                            best_branch=onto,
                            reason=st.reason,
                        )
                        futures[st.slug] = pool.submit(
                            self.dispatch, "tuner", f"rebase-{st.slug}", turn_prompt(req)
                        )
                self._split_turns(camp, pool, futures, budget, spent=spent)
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
                        for r in self._roadmap(n + 1, tried=list(camp.state.candidates))
                        if r.get("slug") not in camp.state.candidates
                    ]
                    continue
                if stopping and not busy:
                    # Tuner turns still out are not waited for: nothing would measure them.
                    break
                if (
                    not futures
                    and not busy
                    and (
                        not roadmap or misses >= max_misses or len(camp.state.candidates) >= budget
                    )
                ):
                    break
                time.sleep(self.poll_s)
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
            futures[variant] = pool.submit(self._ask_variant, st, variant, self._standings())

    def _ask_variant(self, st: CandidateState, variant: str, standings: str) -> str:
        """One tuner turn: ``st``'s change gated on the mode it lost in, at its measured tree."""
        s, base = self.spec, str(st.split["base"])
        wt = open_candidate(
            s.repo, s.workdir / "worktrees", variant, base, prefix=self.branch_prefix
        )
        req = render_split_request(
            self._context(standings), slug=st.slug, variant=variant, worktree=wt, base=base,
            branch=f"{self.branch_prefix}/{variant}", best_branch=f"{self.branch_prefix}/best",
            split=st.split, deltas=st.deltas,
            original=s.workdir / "candidates" / st.slug / "decision.yaml",
        )  # fmt: skip
        return self.dispatch("tuner", variant, turn_prompt(req))

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
            goals.append(f"{s.goal_ratio}x the stock reference")
        if s.goal_pct_of_sol is not None:
            basis = self._goal_ceilings()[1]
            goals.append(f"{s.goal_pct_of_sol:.0%} of SOL" + (f" on {basis}" if basis else ""))
        return " and ".join(goals)

    def _goal_met(self, camp: Campaign) -> bool:
        """The current best meets every goal the spec names, on every headline metric.

        ``ratio_to_reference``: that multiple of the stock reference; ``pct_of_sol``: that fraction
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
            reference = load_measurement(self.data, "t-reference")
            if reference is None or not goal_reached(best, reference, s.goal_ratio):
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
        if self.t.get("stopped") and not self.t.get("kept"):
            self.log("[tune] nothing was kept: no final measurement, the baseline stands")
            final_status = "not run: nothing kept before support completed"
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
            m
            for label in ("t-baseline", "t-reference")
            if (m := load_measurement(self.data, label)) is not None
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
        self.t["phase"] = "done"
        self.checkpoint()
        return "tuned"

    def run(self) -> str | None:
        self.t.pop("failure", None)
        # The wall budget counts from here, across restarts and retries.
        self.t.setdefault("started_at", self.clock())
        if self.attempt > 1:
            self.log(f"[tune] attempt {self.attempt}: jobs queue as `{self.perf.prefix}-<label>`")
        if self.t["phase"] in ("ground", "campaign") and self._stop_requested():
            # Fully supported before or while the campaign ran: it ends with what it kept.
            started = self.t.get("started_at")
            after = (self.clock() - float(started)) / 3600 if started is not None else None
            self.t["ended"] = {
                "by": "support-complete",
                "at": self.clock(),
                "after_h": round(after, 2) if after is not None else None,
            }
            self.t["phase"] = "wrap"
            self.checkpoint()
        # Before anything is measured: no keep gate (and no `gate: off`), no campaign.
        if (
            self.t["phase"] in ("ground", "campaign")
            and self.spec.campaign.get("enabled", True)
            and not self._resolve_keep_gate()
        ):
            return "tune-failed:accuracy"
        if self.t["phase"] == "ground" and not self.smoke():
            return "tune-failed:smoke"
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
    should_stop: Callable[[], bool] = lambda: False,
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
                     log=lambda m: wf.journal.event("tune", m), attempt=attempt,
                     should_stop=should_stop)  # fmt: skip


def run_tune_phase(wf: Any, state: Any, log: Any) -> str | None:
    """``trtllm-modeling-v2 tune``: tune the target the task names, as the checkout has it."""
    stage = _stage_for(wf, state)
    return stage if isinstance(stage, str) else stage.run()


def run_tune_task(
    wf: Any,
    state: Any,
    base_commit: str,
    attempt: int = 1,
    should_stop: Callable[[], bool] = lambda: False,
) -> tuple[str, str]:
    """The execution stage's ``tune`` task: the campaign on ``base_commit``.

    Returns the task's decision (``TUNED`` when the campaign wrapped up) and one line for the
    planner: the best tree and what was kept, or what failed. A resumed run keeps its first pin.
    """
    base = str(state.tune.setdefault("base_commit", base_commit))
    stage = _stage_for(wf, state, base, attempt, should_stop)
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
