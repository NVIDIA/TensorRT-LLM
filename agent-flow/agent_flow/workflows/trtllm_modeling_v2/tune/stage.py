# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tune campaign: perf smoke -> ground truth -> modeling campaign -> wrap-up.

Runs as the planned run's ``tune`` task once the core is correct
(:func:`run_tune_task`, pinned to the commit the target is at when it
starts), or directly by ``trtllm-modeling-v2 tune`` on an already-onboarded
target (:func:`run_tune_phase`). Every phase checkpoints into ``state.tune`` and
is idempotent on resume: a label whose measurement completed (at the same
workload, configs and commit) is never re-measured, a turn whose answer is on
disk (SOL, roadmap, candidate, rebase) is never re-dispatched.

``perf.campaign.max_hours`` is the campaign's wall budget, counted from the
task's first start (ground truth, restarts and retries included). Once it is
spent the campaign issues no new candidate, rebase, member-gated variant or
roadmap; the candidates already issued are measured and judged.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import yaml

from ..checks.gates import checkpoint_remote_code
from ..infra.execution.agent_gate import Priority, turn_priority
from ..infra.vcs import git_checked
from . import sol as sol_mod
from .bench import resolve_config
from .campaign import Campaign, CandidateState, read_decision
from .keep_gate import KeepGate, KeepGateError, resolve_keep_gate
from .measure import (
    INFRA_RCS,
    BenchPart,
    BenchSpec,
    Run,
    SpoolRunner,
    host_controls,
    load_measurement,
    spool_host,
    write_index,
)
from .records import (
    TTFT_TOLERANCE,
    Measurement,
    NoiseFloor,
    PerfLedger,
    format_table,
    goal_reached,
    ground_floors,
)
from .requests import (
    CampaignContext,
    render_rebase_request,
    render_request,
    render_split_request,
    turn_prompt,
)
from .schema import campaign_max_hours, validate_perf_block
from .serving_config import derive_serving_config, write_serving_config
from .worktree import open_candidate

PACKAGE = Path(__file__).resolve().parent.parent
#: What a derived serving config serves: the target, never the stock model. ``bench.py``
#: reads a config that names no mode as ``off``; the stock reference overlays ``off``.
TARGET_SELECTION = {"modeling_v2": "require"}
#: The perf smoke before the baseline: con 1, this many prompts of this many output tokens,
#: one job per world size, each stopped at this wall (``perf.campaign.smoke_wall``).
SMOKE_PROMPTS = 4
SMOKE_OSL = 32
SMOKE_WALL = "45m"


@dataclass
class TuneSpec:
    workdir: Path
    repo: Path
    model: str
    target: str
    identity_configs: list[Path]
    isl: int
    osl: int
    concurrency: list[int]
    reference: str
    spools: list[str]
    dev_spools: list[str]
    env_script: Path
    noise_repeats: int = 2
    campaign: dict = field(default_factory=dict)
    embargo: list[str] = field(default_factory=list)
    sister_engines: list[str] = field(default_factory=list)
    wall: str = "2h"
    #: The perf configs, when ``perf.configs`` names several; every measurement covers all.
    parts: list[BenchPart] = field(default_factory=list)
    #: Stop dispatching once every headline metric is this multiple of the stock reference's.
    goal_ratio: float | None = None
    #: The commit to tune. Empty means the checkout's ``HEAD``, measured in place; a commit is
    #: measured in the campaign's own ``best`` worktree, so commits landing on the checkout
    #: meanwhile (later feature work) never reach a measurement.
    base_commit: str = ""
    #: ``perf.sol``: whether a turn derives the speed-of-light ceilings, and the GPU part to price.
    sol: dict = field(default_factory=lambda: {"enabled": True})
    #: Stop dispatching once every headline metric is this fraction of its SOL ceiling.
    goal_pct_of_sol: float | None = None
    #: The task the campaign serves: the contract its keep gate is (``keep_gate.py``).
    task: dict = field(default_factory=dict)

    @classmethod
    def from_task(cls, task: Mapping[str, Any], workspace: Path, *, target: str = "") -> "TuneSpec":
        perf = task.get("perf") or {}
        workload = perf.get("workload") or {}
        env = task.get("environment") or {}
        session_spools = [s.get("spool") for s in env.get("sessions") or [] if s.get("spool")]
        spools = [str(s) for s in perf.get("spools") or session_spools]
        workdir = Path(workspace).resolve() / "tune"
        isl, osl = int(workload.get("isl", 1024)), int(workload.get("osl", 1024))
        concurrency = [int(c) for c in workload.get("concurrency", [1, 128])]
        identity = [Path(p).resolve() for p in perf.get("identity_configs") or []]
        parts = [_part(task, workdir, entry) for entry in perf.get("configs") or []]
        if parts:
            identity = []
            isl, osl, concurrency = parts[0].isl, parts[0].osl, list(parts[0].concurrencies)
        elif not identity:
            config = {
                **TARGET_SELECTION,
                **derive_serving_config(
                    isl=isl, osl=osl, concurrency=concurrency, draft_len=_draft_len(task)
                ),
            }
            identity = [write_serving_config(workdir / "configs" / "serving.yaml", config)]
        goal = perf.get("goal") if isinstance(perf.get("goal"), Mapping) else {}
        return cls(
            # Absolute: every path in a job script is read on a compute node from
            # the job's own cwd (the tree measured), not from where the CLI ran.
            workdir=workdir,
            repo=Path(task.get("trtllm_repo_path") or ".").resolve(),
            model=str(perf.get("model") or task.get("checkpoint_path")),
            target=str(perf.get("target") or target),
            identity_configs=identity,
            isl=isl,
            osl=osl,
            concurrency=concurrency,
            reference=str(perf.get("reference", "stock")),
            spools=[str(Path(s).resolve()) for s in spools],
            dev_spools=[str(Path(s).resolve()) for s in perf.get("dev_spools") or spools],
            env_script=Path(
                perf.get("env_script") or Path(workspace) / "tune" / "env.sh"
            ).resolve(),
            noise_repeats=int(perf.get("noise_repeats", 2)),
            campaign=dict(perf.get("campaign") or {}),
            embargo=[
                str(e)
                for e in (task.get("source_embargo") or []) + (perf.get("source_embargo") or [])
            ],
            sister_engines=[str(s) for s in perf.get("sister_engines") or []],
            wall=str(perf.get("wall", "2h")),
            parts=parts,
            goal_ratio=float(goal["ratio_to_reference"]) if "ratio_to_reference" in goal else None,
            sol={"enabled": True, **dict(perf.get("sol") or {})},
            goal_pct_of_sol=float(goal["pct_of_sol"]) if "pct_of_sol" in goal else None,
            task=dict(task),
        )

    def describe_workload(self) -> str:
        if not self.parts:
            return (
                f"ISL {self.isl} / OSL {self.osl}, concurrency {self.concurrency}, CUDA graphs on."
            )
        rows = [
            f"`{p.name}`: ISL {p.isl} / OSL {p.osl}, concurrency {list(p.concurrencies)}"
            + (f", constraint {dict(p.constraint)}" if p.constraint else "")
            for p in self.parts
        ]
        return "Every candidate is measured at each perf config: " + "; ".join(rows) + "."

    def bench(self) -> BenchSpec:
        return BenchSpec(
            bench_py=str(PACKAGE / "tune" / "bench.py"), model=self.model, env_script=str(self.env_script),
            isl=self.isl, osl=self.osl, concurrencies=self.concurrency, tree=str(self.repo), wall=self.wall,
            parts=tuple(self.parts),
        )  # fmt: skip


def _draft_len(task: Mapping[str, Any]) -> int:
    """The draft length a speculative target verifies, or 0 without one."""
    from ..checks.gates import resolve_max_draft_len
    from ..run.stages.common import ACCEPTANCE_GATED_UNITS

    speculators = [f for f in task.get("features") or [] if f in ACCEPTANCE_GATED_UNITS]
    if not speculators:
        return 0
    return resolve_max_draft_len(task.get("accuracy") or {}, speculators[-1])


def _part(task: Mapping[str, Any], workdir: Path, entry: Mapping[str, Any]) -> BenchPart:
    """One ``perf.configs`` entry: its workload, and its configs (derived when it names none)."""
    name = str(entry["name"])
    workload = entry.get("workload") or {}
    isl, osl = int(workload.get("isl", 1024)), int(workload.get("osl", 1024))
    concurrency = [int(c) for c in workload.get("concurrency", [1])]
    configs = [Path(p).resolve() for p in entry.get("identity_configs") or []]
    if not configs:
        config = {
            **TARGET_SELECTION,
            **derive_serving_config(
                isl=isl, osl=osl, concurrency=concurrency, draft_len=_draft_len(task)
            ),
        }
        if checkpoint_remote_code(task.get("checkpoint_path")):
            config["trust_remote_code"] = True
        config.update(entry.get("llm_api_options") or {})
        configs = [write_serving_config(workdir / "configs" / f"serving-{name}.yaml", config)]
    return BenchPart(
        name=name,
        isl=isl,
        osl=osl,
        concurrencies=concurrency,
        configs=configs,
        constraint={k: float(v) for k, v in (entry.get("constraint") or {}).items()},
    )


def _world_size(configs: Sequence[Path]) -> int:
    """The ranks one server of these configs runs, as ``bench.py`` launches it."""
    config = resolve_config(list(configs), [])
    world = 1
    for key in ("tensor_parallel_size", "pipeline_parallel_size", "context_parallel_size"):
        world *= int(config.get(key) or 1)
    return world


def _differs(path: Path, bench: BenchSpec, configs: Sequence[Path], tree: str | None) -> str:
    """How the measurement under ``path`` differs from what ``bench`` measures now, or ``""``.

    Read from what ``bench.py`` recorded (``meta.json``, per part): the workload, the merged
    LLM-API config, and the commit when the measurement runs on a tree of the campaign's
    own. A field the measurement did not record, or configs that no longer read, are not
    held against it.
    """
    runs = [
        (path / p.name, p.isl, p.osl, p.concurrencies, [*p.configs, *configs]) for p in bench.parts
    ] or [(path, bench.isl, bench.osl, bench.concurrencies, list(configs))]
    commit = git_checked(Path(tree), "rev-parse", "HEAD", check=False) if tree else ""
    for where, isl, osl, concurrencies, paths in runs:
        try:
            meta = json.loads((where / "meta.json").read_text())
        except (OSError, ValueError):
            return f"has no readable {where.name}/meta.json"
        for key, want in (("isl", isl), ("osl", osl), ("concurrencies", list(concurrencies))):
            if key in meta and meta[key] != want:
                return f"was measured at {key} {meta[key]}, not {want}"
        if "config" in meta:
            try:
                # The same round trip bench.py's meta.json took.
                config = json.loads(json.dumps(resolve_config(paths, []), default=str))
            except (OSError, yaml.YAMLError):
                config = meta["config"]
            if meta["config"] != config:
                return f"was measured at another config than {[p.name for p in paths]}"
        measured = (meta.get("tree") or {}).get("commit")
        if commit and measured and measured != commit:
            return f"was measured at {str(measured)[:10]}, not {commit[:10]}"
    return ""


def _practical(path: Path) -> dict[str, float]:
    """Measured over published peak per bound, from a ``sol.yaml`` that records both.

    ``published_peaks`` and ``measured_peaks`` are keyed by the bound each peak sets
    (``memory``: HBM bandwidth, ``compute``: dense FLOP/s, ``comm``: link bandwidth), in
    one unit per key; a bound missing from either has no factor.
    """
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    if not isinstance(raw, Mapping):
        return {}
    published, measured = raw.get("published_peaks"), raw.get("measured_peaks")
    if not isinstance(published, Mapping) or not isinstance(measured, Mapping):
        return {}
    factors = {}
    for bound in sol_mod.BOUNDS:
        try:
            ratio = float(measured[bound]) / float(published[bound])
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            continue
        if ratio > 0:
            factors[bound] = round(min(ratio, 1.0), 4)
    return factors


def _key(run: Run, labels: Mapping[str, Sequence[Path]]) -> tuple[tuple[str, ...], str]:
    """What ``run`` measures: its label's configs and its member. Repeats share it."""
    return tuple(map(str, labels[run.label])), run.part


def _floor_line(same: NoiseFloor, cross: NoiseFloor) -> str:
    """The ground floors in one journal line: the same-host floor, and the spread across hosts."""

    def widest(floor: NoiseFloor) -> str:
        metric = max(floor.spread, key=floor.spread.get)
        return f"{floor.spread[metric]:.2%} ({metric})"

    line = (
        f"noise floor: same-host spread up to {widest(same)} over {same.samples} runs"
        if same.spread
        else "noise floor: no member ran twice on one node; each node's controls (A, A') set it"
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
    """``perf.campaign.ttft_tolerance``: how much slower a first token may come; null: none judged."""
    if "ttft_tolerance" not in campaign:
        return TTFT_TOLERANCE
    value = campaign["ttft_tolerance"]
    return None if value is None else float(value)


#: Where a wrapped-up campaign lists, in the workspace, what it kept that the target does not carry.
UNMERGED_FILENAME = "UNMERGED_OPTIMIZATIONS.md"


def unmerged_optimizations(ledger: Path, candidates: Path, *, branch: str, best_commit: str) -> str:
    """The campaign's kept changes, one section each, for a human to merge; ``""`` with none.

    Read from the ledger's ``candidate_kept`` records, with each candidate's
    hypothesis from its decision file. Correctness is delivered without them,
    so the target does not carry them: they stay on ``branch``.
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

    ``attempt`` is the task's attempt (1, then +1 per retry of the same task): it is part
    of every job name after the first, so an attempt's receipts are its own. ``clock`` is
    the wall clock the budget is read on. ``should_stop`` says the run no longer needs the
    campaign (full support holds): it then wraps up as when its wall budget is spent,
    and drops what it has not started measuring.
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
        failures: int = 1,
        phase: str = "",
    ) -> dict[str, Measurement | None]:
        """Measure every label not yet on disk, its runs spread over the idle nodes.

        A label is one run per perf config (:meth:`BenchSpec.runs`); each idle node takes
        its share of the runs left as one job (:meth:`_share`). A run on disk is reused only
        when it measured what ``bench`` measures now (:meth:`_reusable`). A run that fails is
        re-queued ``failures`` times, twice when the substrate ended its job; past that its
        label is given up.
        """
        bench = bench or self.spec.bench()
        todo: list[Run] = []
        for label, configs in labels.items():
            if not self._reusable(label, configs, tree, bench):
                todo += [
                    r for r in bench.runs(label) if not self._run_reusable(r, configs, tree, bench)
                ]
        queued = self._adopt_runs(todo, labels)
        retries: dict[str, int] = {}
        while todo:
            held = {run for runs in queued.values() for run in runs}
            waiting = [run for run in todo if run not in held]
            free = self.perf.free_spools() if waiting else []
            while waiting and free:
                # The node most of the waiting members are new to chooses first.
                ran = {sp: self._ran(sp, labels, bench, queued) for sp in free}
                spool = max(
                    free, key=lambda sp: sum(not ran[sp].get(_key(r, labels)) for r in waiting)
                )
                free.remove(spool)
                runs = self._share(waiting, ran[spool], labels, bench, queued)
                job = runs[0].name
                self.perf.submit(spool, job, bench.runs_command(runs, self.data, labels),
                                 wall=bench.wall, tree=tree)  # fmt: skip
                queued[job] = runs
                waiting = [run for run in waiting if run not in runs]
                self.log(f"[tune] measuring {', '.join(r.name for r in runs)} on {spool}")
            self._note_queued(queued)
            for job, spool, rc in self.perf.poll():
                runs = queued.pop(job, [])
                missing = [run for run in runs if not self._measured(run)]
                if missing:
                    # The job's runs go in order: its first missing one is the run that failed
                    # (a session that died, a full filesystem -- or an adopted receipt whose
                    # results are gone, rc 0), and the runs after it never started.
                    failed = missing[0]
                    retries[failed.name] = retries.get(failed.name, 0) + 1
                    limit = 2 if rc in INFRA_RCS else (1 if rc == 0 else failures)
                    if retries[failed.name] <= limit:
                        self.log(f"[tune] {failed.name} failed rc={rc}; re-queued")
                    else:
                        self.log(f"[tune] {failed.name} failed rc={rc}")
                        self.failed[failed.label] = (str(spool), rc, job)
                        todo = [run for run in todo if run.label != failed.label]
                todo = [run for run in todo if run not in runs or run in missing]
            if todo:
                time.sleep(self.poll_s)
        self._note_queued({})
        out = {}
        for label in labels:
            path = self.data / label
            runs = bench.runs(label)
            if bench.parts and not (path / "meta.json").exists() and all(map(self._measured, runs)):
                write_index(self.data, label, bench.parts)
            out[label] = Measurement.load(path) if (path / "meta.json").exists() else None
            if out[label]:
                self.failed.pop(label, None)
                hosts = out[label].meta.get("hosts")
                self.ledger.measurement(out[label], phase=phase or self.t["phase"],
                                        **({"hosts": hosts} if hosts else {}))  # fmt: skip
        return out

    def _measured(self, run: Run) -> bool:
        return (self.data / run.label / run.part / "meta.json").exists()

    def _ran(
        self,
        spool: str,
        labels: Mapping[str, Sequence[Path]],
        bench: BenchSpec,
        queued: Mapping[str, Sequence[Run]],
    ) -> dict[tuple[tuple[str, ...], str], int]:
        """How often ``spool``'s host ran (or runs) each member at each of ``labels``' configs."""
        jobs = {job: sp for sp, job in self.perf.inflight.values()}
        where = {run: jobs[job] for job, runs in queued.items() if job in jobs for run in runs}
        host = spool_host(spool) or spool
        ran: dict[tuple[tuple[str, ...], str], int] = {}
        for label in labels:
            for run in bench.runs(label):
                if run in where:
                    on = spool_host(where[run]) or where[run]
                elif self._measured(run):
                    try:
                        meta = json.loads(
                            (self.data / run.label / run.part / "meta.json").read_text()
                        )
                    except (OSError, ValueError):
                        continue
                    on = str(meta.get("hostname") or "")
                else:
                    continue
                if on == host:
                    ran[_key(run, labels)] = ran.get(_key(run, labels), 0) + 1
        return ran

    def _share(
        self,
        waiting: Sequence[Run],
        ran: Mapping[tuple[tuple[str, ...], str], int],
        labels: Mapping[str, Sequence[Path]],
        bench: BenchSpec,
        queued: Mapping[str, Sequence[Run]],
    ) -> list[Run]:
        """The runs a free node measures next, in one job; ``ran`` is what its host has run.

        Its fair share of the runs not yet measured over the live nodes -- or, while the
        repeats have runs of members its host has not run, one of each (a whole set of
        members on one host is a control of it, :func:`host_controls`). Never more runs than
        one label has: a job stays inside the wall a whole label is given. A node takes first
        the members it has run the fewest times, so a member's repeats spread over the hosts.
        """
        same = [_key(Run(label), labels)[0] for label in labels]
        repeated = {r for r in waiting if same.count(_key(r, labels)[0]) > 1}
        new = {r.part for r in repeated if not ran.get(_key(r, labels))}
        live = sum(1 for sp in self.perf.spools if self.perf.alive(sp)) or 1
        load = len(waiting) + sum(len(runs) for runs in queued.values())
        size = min(len(bench.parts) or 1, max(-(-load // live), len(new)))
        order = {label: n for n, label in enumerate(labels)}
        members = {part.name: n for n, part in enumerate(bench.parts)}
        counts, pool, chosen = dict(ran), list(waiting), []
        while pool and len(chosen) < size:
            run = min(pool, key=lambda r: (counts.get(_key(r, labels), 0), r not in repeated,
                                           members.get(r.part, 0), order[r.label]))  # fmt: skip
            pool.remove(run)
            chosen.append(run)
            counts[_key(run, labels)] = counts.get(_key(run, labels), 0) + 1
        return chosen

    #: ``state.tune`` key: the jobs a measurement has in flight, by job, with their runs.
    QUEUED_RUNS = "queued_runs"

    def _note_queued(self, queued: Mapping[str, Sequence[Run]]) -> None:
        """Checkpoint the jobs in flight and their runs, for a restart to adopt."""
        note = {job: [[run.label, run.part] for run in runs] for job, runs in queued.items()}
        if note != (self.t.get(self.QUEUED_RUNS) or {}):
            self.t[self.QUEUED_RUNS] = note
            self.checkpoint()

    def _adopt_runs(
        self, todo: Sequence[Run], labels: Mapping[str, Sequence[Path]]
    ) -> dict[str, list[Run]]:
        """The jobs an earlier process left in flight for ``todo``, adopted: never run twice.

        Those it checkpointed (:meth:`_note_queued`), and a label's whole job queued before
        its runs were spread.
        """
        queued: dict[str, list[Run]] = {}
        for job, names in (self.t.get(self.QUEUED_RUNS) or {}).items():
            runs = [run for run in todo if [run.label, run.part] in names]
            if runs and self.perf.adopt(job):
                queued[job] = runs
        held = {run for runs in queued.values() for run in runs}
        for label in labels:
            runs = [run for run in todo if run.label == label and run not in held]
            if runs and label not in queued and self.perf.adopt(label):
                queued[label] = runs
        return queued

    def _reusable(
        self, label: str, configs: Sequence[Path], tree: str | None, bench: BenchSpec
    ) -> bool:
        """``label`` is on disk and measured the workload, configs and commit ``bench`` would.

        A measurement of anything else -- the spec or the flow changed between two attempts
        of the task, the best tree moved since -- is set aside under ``<label>.stale-<time>``
        and measured again, never taken as this run's.
        """
        path = self.data / label
        if not (path / "meta.json").exists():
            return False
        why = _differs(path, bench, configs, tree)
        if not why:
            return True
        aside = path.with_name(f"{label}.stale-{int(time.time())}")
        path.rename(aside)
        self.ledger.append("stale_measurement", label=label, reason=why, moved_to=str(aside))
        self.log(f"[tune] {label} on disk {why}; set aside as {aside.name} and measured again")
        return False

    def _run_reusable(
        self, run: Run, configs: Sequence[Path], tree: str | None, bench: BenchSpec
    ) -> bool:
        """A run of a label not yet joined is on disk and measured what ``bench`` would.

        One that measured anything else is set aside under ``<part>.stale-<time>`` and
        measured again (:meth:`_reusable`, for a run).
        """
        path = self.data / run.label
        if not run.part or not (path / run.part / "meta.json").exists():
            return False
        part = next(p for p in bench.parts if p.name == run.part)
        why = _differs(path, replace(bench, parts=(part,)), configs, tree)
        if not why:
            return True
        aside = path / f"{run.part}.stale-{int(time.time())}"
        (path / run.part).rename(aside)
        self.ledger.append("stale_measurement", label=run.name, reason=why, moved_to=str(aside))
        self.log(f"[tune] {run.name} on disk {why}; set aside as {aside.name} and measured again")
        return False

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
        """The first perf config of each world size the campaign serves: ``{world: (name, configs)}``."""
        s = self.spec
        runs = [(p.name, list(p.configs)) for p in s.parts] or [("identity", s.identity_configs)]
        out: dict[int, tuple[str, list[Path]]] = {}
        for name, configs in runs:
            out.setdefault(_world_size(configs), (name, list(configs)))
        return dict(sorted(out.items()))

    def smoke(self) -> bool:
        """The perf preflight, before the baseline: one short serve per world size.

        Con 1, ``SMOKE_PROMPTS`` prompts of ``SMOKE_OSL`` tokens, on the perf sessions and the
        tree the baseline measures. Only the substrate's own failures are re-queued. Skipped
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
        got = self._measure(labels, tree=tree, bench=bench, failures=0, phase="smoke")
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
        if s.noise_repeats < 2:
            self.log(
                f"[tune] noise_repeats={s.noise_repeats}: the noise floor rests on {s.noise_repeats + 1} samples "
                "and under-estimates the same-host range; small differences will not be resolvable"
            )
        tree = str(self._best_tree()) if s.base_commit else None
        got = self._measure(labels, tree=tree)
        if got.get("t-baseline") is None:
            self.t["failure"] = f"baseline: {self._failure('t-baseline')}"
            self.log(
                "[tune] the baseline could not be measured; the campaign cannot start "
                f"({self.t['failure']})"
            )
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
            standings_table=standings, sister_engines=s.sister_engines, notes=self._sources_note(),
        )  # fmt: skip

    def _sources_note(self) -> str:
        """Where a tuner looks before writing a kernel."""
        s = self.spec
        surface = s.workdir.parent / "surface.md"
        lines = ["## Sources to check before writing a kernel", ""]
        if surface.exists():
            lines.append(
                f"- FlashInfer **as installed**: `{surface}` section 'FlashInfer API' lists the installed version's "
                "public callables the engine does not call yet -- a demand one answers is a tier-3 entry "
                "(`from flashinfer import ...` in the entry), not a new kernel."
            )
        lines += [
            f"- Earlier choices for this target: `{s.workdir / '08-kernel-sources' / 'kernel_choices.json'}` (chosen + "
            "alternatives with timings); extend it, do not re-race what is recorded.",
            f"- Prior art in this repository: `git -C {s.repo} log --oneline -i --grep='perf' -- "
            "tensorrt_llm/_torch/modules/<similar module> tensorrt_llm/_torch/models/<similar model>` (strategies "
            "for the same op shapes in other models -- never the embargoed in-tree implementation of this model).",
            "- Profiles of the current best tree: the latest `roadmap*.yaml` evidence in the campaign workdir.",
        ]
        return "\n".join(lines)

    def _roadmap_paths(self) -> str:
        """What a roadmap turn needs to profile the best tree without asking.

        The same facts a candidate request carries, and the embargo.
        """
        s = self.spec
        configs = list(s.identity_configs)
        rows = [
            ("best tree to profile", self.spec.workdir / "worktrees" / "best"),
            ("configs, applied in order", ", ".join(f"`{c}`" for c in configs)),
            ("model", s.model),
            ("dev spools (1-GPU jobs: nsys captures)", ", ".join(f"`{p}`" for p in s.dev_spools)),
            ("env script every job sources", s.env_script),
            ("package (bench.py --nsys, spoolctl)", PACKAGE),
        ]
        lines = ["## Paths", "", "| name | value |", "|---|---|"]
        lines += [f"| {k} | {v if str(v).startswith('`') else f'`{v}`'} |" for k, v in rows]
        lines += [
            "",
            "## Embargo (do not read, grep or call)",
            "",
            *(f"- `{e}`" for e in s.embargo),
        ]
        return "\n".join(lines)

    def _sisters_line(self) -> str:
        if not self.spec.sister_engines:
            return ""
        return (
            "\n- Other engines that implement this model (strategies, not code): "
            + ", ".join(f"`{e}`" for e in self.spec.sister_engines)
            + "."
        )

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

    def _sol_request(self, baseline: Measurement, problems: Sequence[str] = ()) -> Path:
        s = self.spec
        sol_dir = s.workdir / "sol"
        sol_dir.mkdir(parents=True, exist_ok=True)
        what = "sol" if not problems else "sol-fix"
        request = s.workdir / f"{what}-request.md"
        gpu = str(s.sol.get("gpu") or "")
        keys = "\n".join(f"  {k}: {{value: <ceiling>, bound: <{'|'.join(sol_mod.BOUNDS)}>, "
                         f"derivation: <formula with numbers>}}" for k in baseline.headline())  # fmt: skip
        fix = ""
        if problems:
            fix = (
                "\n\n## Your last `sol.yaml` was refused\n\n"
                + "\n".join(f"- {p}" for p in problems)
                + "\n\nFix the derivation, not the number: a ceiling below a measurement means a "
                "byte, a FLOP or a peak was counted wrong.\n"
            )
        request.write_text(
            f"# Tuner request: {what}\n\n"
            "Derive the speed-of-light (SOL) ceiling of every headline metric this campaign "
            "measures. The roadmap ranks work by the headroom each region has to its ceiling, "
            "and the campaign reports how far the target is from it. You change no code.\n\n"
            f"1. Load the `{sol_mod.SOL_SKILL}` skill (`{sol_mod.SOL_SKILL_QUALIFIED}` if the bare "
            "name is not found); it is the methodology every number comes from. Resolve the "
            "hardware peaks with its peaks calculator"
            + (f" (`--part {gpu}`)" if gpu else " (the part `nvidia-smi` names in a spool job)")
            + ", never from memory. Measure its latency constants with its `measure_channels.py` "
            "as a spool job on a compute session (`spoolctl.py ... launch`; `launch --pick` "
            "takes the session of that kind that can start it soonest), never on this host, "
            f"and merge them into `{sol_dir / 'peaks.json'}`.\n"
            f"2. Read the checkpoint's `config.json` (`{s.model}`) and the target's forward "
            f"(`{s.workdir / 'worktrees' / 'best'}`, under `{s.target}`) for what one step "
            "necessarily does: the weight and KV bytes it reads and writes, its FLOPs per token, "
            "its collectives at this parallel size. Count necessary work, not what the current "
            "kernels happen to do.\n"
            "3. Per perf config, price one decode step at the lowest concurrency (the "
            "`tok_s_user` ceiling) and the steady state at the highest (the `output_tok_s_gpu` "
            "ceiling) with the skill's calculator, and write every formula with its numbers.\n\n"
            f"Write `{s.workdir / sol_mod.SOL_FILENAME}`:\n\n"
            "```yaml\nmethod: calculator\n"
            f"peaks: {sol_dir / 'peaks.json'}\nceilings:\n{keys}\nnotes: <assumptions>\n```\n\n"
            "If the skill does not load, derive the ceilings by hand from published peaks, name "
            "each peak's source in `derivation`, and write `method: reduced` with no `peaks`. "
            "With `method: reduced`, also record the peaks keyed by the bound each sets, one "
            "unit per key: `published_peaks: {memory: <HBM TB/s>, compute: <dense TFLOP/s>, "
            "comm: <link GB/s>}`, and `measured_peaks` with the same keys for every peak you "
            "measured in a spool job. A `pct_of_sol` goal is judged against the reduced "
            "ceilings scaled by measured / published for each ceiling's bound. "
            "Then write the decision file `"
            + str(s.workdir / f"{what}-decision.yaml")
            + "` with `decision: CANDIDATE` (written) or `STOP` (no defensible ceiling; say why "
            "in `notes`).\n\n## Standings\n\n" + self._standings() + fix + "\n"
        )
        return request

    def _sol(self) -> None:
        """Ask for the SOL ceilings once, after ground truth; one fix turn, then go on without.

        A turn that finished while no orchestrator watched it (a restart) is not asked again:
        a ``sol.yaml`` that loads, or the turn's decision file, is its answer.
        """
        if not self.spec.sol.get("enabled", True) or "sol" in self.t:
            return
        baseline = self._baseline()
        if baseline is None:
            return
        path = self.spec.workdir / sol_mod.SOL_FILENAME
        ceilings, errors = sol_mod.load_sol(path, baseline, self.spec.workdir)
        if ceilings is None and not self._answered("sol"):
            self.dispatch("tuner", "sol", turn_prompt(self._sol_request(baseline)))
            ceilings, errors = sol_mod.load_sol(path, baseline, self.spec.workdir)
        if errors and path.exists() and not self._answered("sol-fix"):
            self.dispatch("tuner", "sol-fix", turn_prompt(self._sol_request(baseline, errors)))
            ceilings, errors = sol_mod.load_sol(path, baseline, self.spec.workdir)
        if ceilings is None:
            self.log(
                f"[tune] no SOL ceilings; the campaign proceeds without them ({errors[0][:240]})"
            )
            self.t["sol"] = {"status": "unavailable", "errors": errors}
            self.ledger.append("sol", status="unavailable", errors=errors)
        else:
            practical = {} if ceilings.calculated else _practical(path)
            self.t["sol"] = {
                **sol_mod.as_state(ceilings),
                **({"practical": practical} if practical else {}),
            }
            pct = ceilings.pct_of_sol(baseline)
            self.ledger.append("sol", status=ceilings.method, pct_of_sol=pct, practical=practical)
            self.log(f"[tune] SOL ({ceilings.method}): the baseline is at {pct} % of SOL")
        self.checkpoint()

    def _sol_ask(self, what: str) -> str:
        """With ceilings, the roadmap ranks by the headroom each region has to them."""
        ceilings = self._ceilings()
        if ceilings is None:
            return ""
        peaks = f"with the peaks file `{ceilings.peaks}`" if ceilings.calculated else "by hand"
        return (
            f"\n\nRank by headroom, not by time. Join the profile's regions (op type x shape "
            f"class) against their SOL with the `{sol_mod.SOL_SKILL}` skill's `sol_calc.py "
            f"analyze` {peaks}, write the joined table to `{self.spec.workdir / f'{what}-sol.json'}`, "
            "and order the roadmap by the gap (measured - SOL) each item can recover. A region "
            "near its ceiling can still be eliminated, fused or overlapped, never made faster: "
            "a kernel ledger `faster` dismissal as at-sol-floor cites that region's % of SOL."
        )

    def _roadmap(self, n: int = 1, tried: Sequence[str] = (), ask: bool = True) -> list[dict]:
        """Roadmap ``n``: the first from the baseline profile, later ones re-profiled on the moved best tree.

        With ``ask=False`` (the wall budget is spent) no turn is asked: the roadmap is what is on disk.
        """
        what = "roadmap" if n == 1 else f"roadmap-{n}"
        path = self.spec.workdir / f"{what}.yaml"
        if not ask:
            return (
                [dict(r) for r in yaml.safe_load(path.read_text()) or []] if path.exists() else []
            )
        if not path.exists() and not self._answered(what):
            request = self.spec.workdir / f"{what}-request.md"
            again = ""
            if tried:
                again = (
                    "\n\nThis is roadmap "
                    + str(n)
                    + ": re-profile the best tree (`"
                    + str(self.spec.workdir / "worktrees" / "best")
                    + "`) -- the kept candidates moved the profile. Already attempted (do not repeat; "
                    "their decision files under `"
                    + str(self.spec.workdir / "candidates")
                    + "/<slug>/decision.yaml` list measured follow-ups worth ranking): "
                    + ", ".join(f"`{t}`" for t in tried)
                    + "."
                )
            request.write_text(
                f"# Tuner request: {what}\n\nProfile the target at the campaign config (nsys through the substrate: "
                f"`{PACKAGE}/tune/bench.py --nsys` in a spool job, read with "
                "`nsys stats --report cuda_gpu_kern_sum` on its `.nsys-rep`), read the forward, and "
                f"write `{path}`: a YAML list of "
                "`{slug, kind, hypothesis, evidence, expected, modeling_edits, sources}` ranked by the "
                "exclusive time each can remove; `sources` names where the idea and its kernel options "
                "come from (`profile`, `code-reading`, `flashinfer`, `repo-prior-art`, `sister-engine`, "
                "`kernel-choices`, each with a reference). Then write the "
                "decision file `"
                + str(self.spec.workdir / f"{what}-decision.yaml")
                + "` with `decision: CANDIDATE` (roadmap ready) or `STOP`."
                + self._sol_ask(what)
                + self._ledger_ask(what)
                + again
                + "\n\n"
                + self._roadmap_paths()
                + "\n\n## Standings\n\n"
                + self._standings()
                + "\n\n"
                + self._sources_note()
                + self._sisters_line()
                + "\n"
            )
            self.dispatch("tuner", what, turn_prompt(request))
        if not path.exists():
            return []
        items = [dict(r) for r in yaml.safe_load(path.read_text()) or []]
        if items and self._ledger_mode() != "off" and not self.t.get(f"{what}_ledger_checked"):
            errors = self._check_ledger(what, items)
            if errors:
                self.log(
                    f"[tune] {what}: kernel ledger incomplete; the campaign proceeds ({errors[0][:240]})"
                )
                self.ledger.append("kernel_ledger", roadmap=what, errors=errors)
            self.t[f"{what}_ledger_checked"] = True
            self.checkpoint()
        return items

    def _ledger_mode(self) -> str:
        return str(self.spec.campaign.get("kernel_ledger", "require"))

    def _ledger_path(self, what: str) -> Path:
        return self.spec.workdir / f"{what}-kernel_ledger.yaml"

    def _ledger_ask(self, what: str) -> str:
        """perf_optimize's kernel ledger, asked of every roadmap: the exhaustiveness proof behind it."""
        if self._ledger_mode() == "off":
            return ""
        from agent_flow.workflows.perf_optimize import kernel_ledger

        target = float(self.spec.campaign.get("kernel_coverage_pct", 80.0))
        return (
            f"\n\nAlso write `{self._ledger_path(what)}`: the agent-flow perf_optimize **kernel ledger** (schema and "
            f"example in the docstring of `{kernel_ledger.__file__}`). One row per kernel (or kernel family) of the "
            f"best tree's decode profile, down the ranking until the rows cover >= {target:g}% of GPU time "
            "(`nsys stats --report cuda_gpu_kern_sum`); each row answers elimination / faster / fusion / "
            "overlap with a roadmap slug (`disposition: item`) or an evidence-backed dismissal. Without an "
            "ncu pass, write "
            '`ncu: "unavailable: nsys-only roadmap turn"` and the estimated `bound` on the row. The stage '
            "validates it with perf_optimize's own `load_ledger` + `cross_validate`."
        )

    def _check_ledger(self, what: str, items: Sequence[Mapping]) -> list[str]:
        """Validate the roadmap's kernel ledger with perf_optimize's validator; returns the problems."""
        from agent_flow.workflows.perf_optimize.kernel_ledger import (
            QUESTIONS,
            LedgerError,
            cross_validate,
            load_ledger,
        )

        try:
            ledger = load_ledger(self._ledger_path(what))
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
        """Resolve the keep gate from the contract; False, with the failure recorded, if it cannot be."""
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

        The first look that finds it spent is journaled and goes to the ledger
        (``budget_spent``). A budget raised since then is not spent.
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
        """Record why the campaign ended: goal, full support, budget, misses, candidates, roadmaps."""
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
            # A campaign that ended and is entered again (an explicit `tune`) has a budget of its own.
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
                            read_decision(done)
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
                        # The worktree checkout (minutes on a parallel filesystem) runs in the turn's
                        # own thread, so turns start side by side instead of one checkout at a time.
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
                            read_decision(decision)
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
                                # One variant per split: a turn that answered nothing is not asked again.
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
                    # One rebase per tree it conflicted with: a winner left out of a batch is
                    # rebased onto the batch (pinned to its commit) while the batch is measured.
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
                        # A textual conflict with an earlier keep is not a verdict: rebase and retry.
                        self.t[key] = True
                        onto = (
                            st.rebase_ref
                            if st.rebase_onto == "tune/eval-batch"
                            else f"{self.branch_prefix}/best"
                        )
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
                    # The roadmap is spent with budget left: profile the moved best tree for the next one.
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

        None is asked for once the budget of candidates or the wall budget (``spent``) is
        spent, or the goal reached: the split is reverted. A variant whose turn a restart cut
        off is submitted when its decision is on disk, and asked again otherwise, unless the
        wall budget is spent.
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
                    read_decision(done)
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
        """One tuner turn: ``st``'s change gated on the mode it lost in, in a worktree at its measured tree."""
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

        A turn that finished updated the candidate's decision with ``rebased_onto`` the
        commit it was asked to rebase onto: the candidate is submitted again. Any other is
        asked again.
        """
        for st in camp.state.candidates.values():
            key = f"rebase-{st.slug}-{st.rebase_ref[:10]}"
            if st.status != "conflict" or not self.t.get(key):
                continue
            decision = self.spec.workdir / "candidates" / st.slug / "decision.yaml"
            try:
                if _same_commit(
                    str(read_decision(decision).get("rebased_onto") or ""), st.rebase_ref
                ):
                    camp.submit_candidate(st.slug, decision)
                    self.log(f"[tune] {st.slug}: its rebase finished during a restart; submitted")
                    continue
            except (OSError, ValueError) as exc:
                self.log(f"[tune] {st.slug}: no valid rebase decision ({exc})")
            self.log(f"[tune] {st.slug}: its rebase was cut off by a restart; asked again")
            del self.t[key]
        self.checkpoint()

    def _best_measurement(self, camp: Campaign) -> Measurement | None:
        """The measurement of the current best: the last keep's, else the baseline."""
        kept = [camp.state.candidates[k] for k in camp.state.kept if k in camp.state.candidates]
        label = next((c.e2e_label for c in reversed(kept) if c.e2e_label), "t-baseline")
        return load_measurement(self.data, label)

    def _goal_ceilings(self) -> "tuple[sol_mod.SolCeilings | None, str]":
        """The ceilings a ``pct_of_sol`` goal is judged against, and that basis in words.

        Calculator ceilings as they are. Reduced (hand-derived) ceilings price published
        peaks a part does not reach in practice; when ``sol.yaml`` recorded measured peaks
        too, each ceiling is scaled by measured / published for its bound (``practical`` in
        the SOL state), else taken at the published peaks.
        """
        ceilings = self._ceilings()
        if ceilings is None:
            return None, ""
        if ceilings.calculated:
            return ceilings, "calculator ceilings"
        practical = {
            str(k): float(v) for k, v in ((self.t.get("sol") or {}).get("practical") or {}).items()
        }
        if not practical:
            return ceilings, "reduced ceilings at published peaks"
        scaled = {
            key: sol_mod.Ceiling(c.value * practical.get(c.bound, 1.0), c.bound, c.derivation)
            for key, c in ceilings.ceilings.items()
        }
        factors = ", ".join(f"{bound} x{f:.3f}" for bound, f in sorted(practical.items()))
        return replace(ceilings, ceilings=scaled), (
            f"reduced ceilings scaled by measured / published peak ({factors})"
        )

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
        """The current best meets every goal the spec names.

        ``ratio_to_reference``: at least that multiple of the stock reference
        on every headline metric. ``pct_of_sol``: at least that fraction of
        every headline's SOL ceiling, on the ceilings :meth:`_goal_ceilings`
        names -- calculator ones, or reduced ones at their practical peaks.
        """
        s = self.spec
        if s.goal_ratio is None and s.goal_pct_of_sol is None:
            return False
        best = self._best_measurement(camp)
        if best is None:
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
        """Record the goal as reached, with the measurement and the ceiling basis that decided it."""
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
        if self.t.get("stopped") and not self.t.get("kept"):
            # Wrapped up with nothing kept: the best tree is the baseline, measured already.
            self.log("[tune] nothing was kept: no final measurement, the baseline stands")
        else:
            self._measure(labels, tree=self.t.get("best_tree") or None)
        rows = [Measurement.load(self.data / label)
                for label in ("t-baseline", "t-reference", "t-final")
                if (self.data / label / "meta.json").exists()]  # fmt: skip
        summary = s.workdir / "summary.md"
        summary.write_text(
            "# Tune stage summary\n\n"
            + format_table(rows, rows[0] if rows else None)
            + f"\n\nModeling kept: {self.t.get('kept', [])}\n"
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
    """The planned run's ``tune`` task: the campaign on ``base_commit``.

    Returns the task's decision (``TUNED`` when the campaign wrapped up) and one line
    for the planner: where the best tree is and what was kept, or what failed and the
    job log that says why. A resumed run keeps the commit it first pinned.
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
