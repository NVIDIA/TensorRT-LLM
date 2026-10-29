# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The graph-driven run loop.

The planner writes the task graph; this loop admits it, starts tasks as
devices and paths allow, runs each finished task's checks itself, and asks
the planner again when something happens. The run ends when the contract
holds, when the planner escalates or hits a ceiling, or when it stalls.

Each task's worker runs in a thread of its own, every check on one bounded
pool, and the planner's turn in a thread beside them, one turn at a time.
Only the loop's own thread changes the graph or writes it: a planner's
revision is applied when its turn returns, against the graph as it is by
then. Work whose shape no planner turn decides (the survey's kernel tasks,
regression checks, promotes) the loop adds itself.

The workflow's services arrive as callables (:class:`Services`), so the loop
runs under fakes on CPU exactly as it runs under the workflow.
"""

from __future__ import annotations

import contextlib
import copy
import errno
import json
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from ..agents import decisions as decision_table
from ..agents.slots import AgentGate, Priority, run_at, turn_priority
from ..infra.files import write_text_atomic
from ..operators import coverage as vocab_mod
from . import graph as graph_mod
from .records import metrics as metrics_mod
from .validation import contract as contract_mod
from .validation.completion import Completion, assess
from .workspace.integration import INTEGRATED_KINDS, TaskNotStarted

#: The planner decisions the loop branches on; any other is a graph update.
DONE, ESCALATE, CEILING = "DONE", "ESCALATE", "CEILING"
#: How many times a refused graph goes back to the planner before the run stops.
MAX_REFUSALS = 3
#: How many planner turns in a row may raise (a timed-out or crashed turn, an
#: unreadable proposal) before the run stops; each waits longer before the next.
MAX_PLANNER_FAILURES = 5
#: The wait before the planner is asked again after its turn raised, doubled per failure.
PLANNER_RETRY_S = 60.0
#: What happened since the planner's last turn, kept across a restart.
EVENTS_FILENAME = "planner_events.json"
#: How many tasks run at once, and how many checks, unless the caller says otherwise.
DEFAULT_MAX_PARALLEL_TASKS = 9
DEFAULT_MAX_PARALLEL_CHECKS = 8
#: Task kinds that run for hours beside the graph: while only they run, the
#: planner reads its events at once instead of waiting for the event window.
BACKGROUND_KINDS = graph_mod.BACKGROUND_KINDS
#: How many times a task the run's infrastructure failed starts again on its own.
INFRA_RETRIES = 1
#: A state write that fails with one of these waits for room instead of ending the run.
STORAGE_FULL = frozenset({errno.EDQUOT, errno.ENOSPC})
#: How many stale checks run again, in the contract's order, beside running work each
#: time the target changes; the rest wait for the graph to run out of work.
RECHECKS_WHILE_BUSY = 4
#: How long a task thread waiting on its soft dependencies sleeps between looks,
#: past the wake-up each recorded outcome gives it.
SOFT_POLL_S = 5.0
#: Per workspace, the stop a run this process drives there answers (:func:`stoppable`).
_STOPS: dict[Path, Future] = {}
_STOPS_LOCK = threading.Lock()


@dataclass
class PlannerTurn:
    """What the planner is told: the contract's standing and what happened since its last turn."""

    evaluation: contract_mod.Evaluation
    events: list[str]
    refusals: list[str] = field(default_factory=list)
    #: The state of the target ``evaluation`` was read on.
    marker: str = ""
    #: Where the regression checks stand: the batch the next one takes, and the
    #: checks running and waiting. Empty when the run has no shared code to check.
    regression: str = ""


@dataclass
class PlannerAnswer:
    decision: str
    summary: str = ""
    graph: graph_mod.Graph | None = None
    #: Why a GRAPH_UPDATED carries no graph: the proposal could not be read.
    problem: str = ""


@dataclass(frozen=True)
class _Outcome:
    """How one run of a task ended, as an event reports it."""

    task: str
    attempt: int
    #: The status the run left the task in: ``done``, ``failed`` or ``abandoned``.
    status: str


@dataclass(frozen=True)
class Event:
    """One planner event: words and scheduling metadata cannot drift apart."""

    text: str

    #: It wakes the planner at once.
    urgent: bool = True
    #: The task run it reports the end of; ``None`` for any other event.
    outcome: _Outcome | None = None
    #: It is a worker's report (:attr:`GraphOrchestrator.unread_reports`).
    report: bool = False
    #: It wakes no turn of its own: a success the graph already answered
    #: (:meth:`GraphOrchestrator._answered_ahead`). The next turn reads it.
    quiet: bool = False
    #: The tasks the orchestrator added that it reports: a revision written from
    #: a graph that holds them all answers it (:meth:`GraphOrchestrator._drop_answered`).
    added: tuple[str, ...] = ()


@dataclass
class Services:
    """The workflow, as the loop needs it."""

    #: One planner turn; returns its decision and, for GRAPH_UPDATED, the graph it wrote
    #: (blocking; runs in a thread of its own, one turn at a time).
    plan: Callable[[PlannerTurn], PlannerAnswer]
    #: One worker for a task; returns its decision (blocking; runs in a thread). Lines after
    #: the first are a note the planner sees with the task's event.
    work: Callable[[graph_mod.Task], str]
    #: Run one contract check now and return its gate result (``name``, ``member``, ``passed``)
    #: (blocking; several run at once, each in a thread).
    check: Callable[[contract_mod.Check], Any]
    #: Devices free for a new task.
    free_gpus: Callable[[], int]
    #: What a task holds of those devices while it runs (``execution.held_gpus``).
    held_gpus: Callable[[graph_mod.Task], int] = lambda task: task.gpus
    checkpoint: Callable[[], None] = lambda: None
    log: Callable[[str], None] = lambda message: None
    #: The target's identity now; a verdict recorded under another is stale.
    marker: Callable[[], str] = lambda: ""
    #: The loop has recorded how a task's turn ended (done, failed, abandoned, or pending
    #: again after a failure of the infrastructure): its answer is consumed.
    finished: Callable[[graph_mod.Task], None] = lambda task: None
    #: Whether a task whose turn did not finish landed commits on the target all the
    #: same, as a failed receipt's guard tests do; one that finished always has.
    landed: Callable[[graph_mod.Task], bool] = lambda task: False
    #: Every required check holds on the target as it is (its marker, and the evaluation):
    #: name the commit it is at, journal and record the milestone, and return it; ``None``
    #: when no commit holds what the checks measured. ``None``: no milestone is kept.
    correct: Callable[[str, contract_mod.Evaluation], contract_mod.Milestone | None] | None = None
    #: The target's local variants of shared entries, given the graph: a
    #: ``promotion.Promotion`` naming the entries waiting, and the promote task to add
    #: now or why none is (``operators/promotion.py``). ``None``: nothing is promoted.
    promotion: Callable[[graph_mod.Graph], Any] | None = None
    #: The contract evaluated on the target (its marker, and the evaluation), before
    #: ``correct``: name the bring-up's commit the first time its checks hold. ``None``:
    #: no bring-up is named.
    bringup: Callable[[str, contract_mod.Evaluation], Any] | None = None
    #: The same, for each cell (a unit or variant at one member) whose checks all hold:
    #: record where. ``None``: no cell is recorded.
    cells: Callable[[str, contract_mod.Evaluation], Any] | None = None
    #: The delivery checks on disk (the CI accuracy test, the test list entry, the
    #: accuracy reference, the target's files and commits), as cheap as reading files:
    #: what fails, one line each; empty when all pass. ``None``: none are checked here.
    delivery_files: Callable[[], list[str]] | None = None
    #: Tell a running campaign to wrap up: full support holds, so it dispatches nothing
    #: new and ends with its best kept tree. Given why.
    stop_campaign: Callable[[str], None] = lambda reason: None
    #: What the stage has for the planner outside any task since last asked (the
    #: operator's messages): each an event of the next turn.
    stage_notes: Callable[[], list[str]] | None = None
    #: What a gate result read of the tree, as the contract records it
    #: (:attr:`~.contract.Verdict.reads`, ``reads_marker``).
    narrow: Callable[[Any], tuple[str, str]] | None = None
    #: Whether a verdict recorded on another state of the target holds on it as it
    #: is now: what it read of the tree is unchanged.
    holds: Callable[[contract_mod.Verdict], bool] | None = None
    #: One regression check (no agent turn): run the package tests after its batch of
    #: changes to shared code and return what failed, one line each (blocking; on the check
    #: pool). ``None``: a regression check passes with nothing run.
    regression: Callable[[graph_mod.Task], list[str]] | None = None


#: What a ``check`` task ends with once its checks are in; no worker writes it.
CHECKED = "CHECKED"
#: What a ``tune`` task ends with once its campaign wrapped up (``tune/stage.py``).
TUNED = "TUNED"
#: Decisions that mean the task did what it was asked: the roles' that finish
#: a task in the decision table (``agents/decisions.py``), and the two the run
#: writes itself.
FINISHED = decision_table.finishing() | {TUNED, CHECKED}


def _not_the_task(decision: str) -> bool:
    """A decision that says the run's infrastructure failed, not the task's work.

    The worker's turn raised, its finished branch could not be integrated, or
    a tune campaign could not measure its baseline.
    """
    return (
        decision.startswith("raised ")
        or decision == "INTEGRATION_FAILED"
        or (decision.startswith("TUNE_FAILED") and "tune-failed:baseline" in decision)
    )


class ChecksRaised(Exception):
    """Some checks raised instead of returning a verdict (a session was lost, say).

    Their outcome is unknown, not a failure: ``failed`` are the keys that did
    fail, ``raised`` what each of the others raised.
    """

    def __init__(self, failed: list[str], raised: Mapping[str, str]) -> None:
        super().__init__("; ".join(f"{key}: {why}" for key, why in raised.items()))
        self.failed = failed


def _failed(submitted: Sequence[tuple[str, Future]]) -> list[str]:
    """Once every submitted check is in, the keys of those that failed.

    A check that raised has no verdict: once all are in, :class:`ChecksRaised`
    names them beside those that failed.
    """
    wait([future for _, future in submitted])
    failed: list[str] = []
    raised: dict[str, str] = {}
    for key, future in submitted:
        try:
            if not future.result():
                failed.append(key)
        except Exception as exc:  # collected: the caller decides what an unknown outcome costs
            raised[key] = f"{type(exc).__name__}: {exc}"
    if raised:
        raise ChecksRaised(failed, raised)
    return failed


class GraphOrchestrator:
    def __init__(
        self,
        services: Services,
        contract: contract_mod.Contract,
        status: contract_mod.ContractStatus,
        workspace: Path,
        *,
        revision_cap: int = 5,
        max_parallel: int = DEFAULT_MAX_PARALLEL_TASKS,
        max_parallel_checks: int = DEFAULT_MAX_PARALLEL_CHECKS,
        poll_s: float = 5.0,
        event_window_s: float = 0.0,
        allowed_paths: Iterable[str] = (),
        max_gpus: int | None = None,
        required_kinds: Iterable[str] = (),
        start_after: Mapping[str, Sequence[str]] | None = None,
        own_paths: Iterable[str] = (),
        repo_root: str | Path = "",
    ) -> None:
        self.s = services
        self.contract = contract
        #: The full benchmarks, which run again at milestones only (:meth:`_settle`).
        self._accuracy_keys = {c.key for c in contract.checks if c.gate.startswith("accuracy[")}
        #: State writes a full filesystem refused, by what they write: each is tried
        #: again as the loop goes round, and nothing new starts until all have landed.
        self._unwritten: dict[str, Callable[[], None]] = {}
        self.status = status
        self.graph_path = Path(workspace) / graph_mod.GRAPH_FILENAME
        self.graph = (
            graph_mod.Graph.load(self.graph_path) if self.graph_path.exists() else graph_mod.Graph()
        )
        # A task recorded as running belonged to a process that is gone; it
        # starts again, and every worker resumes from what it left on disk.
        restarted = [t for t in self.graph.tasks if t.status == "running"]
        for task in restarted:
            task.status = "pending"
        self.workspace = Path(workspace)
        #: Tasks that were running when the last process stopped and have not
        #: started again: they start ahead of every other ready task.
        self._resumed = {t.id for t in restarted}
        if restarted:
            # On disk too: the planner reads graph.yaml, and a task it sees
            # running is one it believes a worker has.
            self._save_graph()
            services.log(
                "[resume] the tasks running when the last process stopped start again: "
                + ", ".join(t.id for t in restarted)
            )
        #: Per kind, the checks that must have passed before a task of that
        #: kind may start.
        self.start_after = {k: tuple(v) for k, v in (start_after or {}).items()}
        self.revision_cap = revision_cap
        self.max_parallel = max_parallel
        #: Checks running at once, across every task and the re-checks of an idle run.
        self.max_parallel_checks = max(1, max_parallel_checks)
        #: How long the loop waits between looks while ready work waits for devices.
        self.poll_s = poll_s
        #: While anything is in flight, events that need no judgement wait up to
        #: this long to reach the planner together.
        self.event_window_s = event_window_s
        self._first_event_at: float | None = None
        self._status_lock = threading.Lock()
        #: The checkout a kernel task's files are named from: a multi-rank entry's op matrix.
        self.repo_root = str(repo_root)
        self.validation = {
            "required_checks": [c.key for c in contract.required()],
            "allowed_paths": tuple(allowed_paths),
            "max_gpus": max_gpus,
            "required_kinds": tuple(required_kinds),
            "own_paths": tuple(own_paths),
            "known_checks": [c.key for c in contract.checks],
        }
        #: Events the planner has not read yet. On disk, so a process that dies
        #: while they wait (or during the planner turn that reads them) hands
        #: them to the next one instead of dropping a report.
        self.events_path = Path(workspace) / EVENTS_FILENAME
        saved: dict[str, Any] = {}
        if self.events_path.is_file():
            saved = json.loads(self.events_path.read_text(encoding="utf-8") or "{}")
        self.events = [Event(str(e)) for e in saved.get("events") or ()]
        #: A worker reported something the planner has not read yet. The run is
        #: not delivered past it: a tune task's best branch may still need merging.
        self.unread_reports = bool(saved.get("unread_reports"))
        #: Per task, how many times it started again after the infrastructure failed it.
        self.retries: dict[str, int] = {
            str(k): int(v) for k, v in (saved.get("retries") or {}).items()
        }
        #: Per task, the attempts whose worktree could not be prepared: no worker ran,
        #: so they keep their numbers but do not count against ``graph.MAX_ATTEMPTS``.
        self.not_started_attempts: dict[str, list[int]] = {
            str(k): sorted({int(n) for n in v})
            for k, v in (saved.get("not_started_attempts") or {}).items()
        }
        #: Per task whose commits landed on the target, when its latest landing was (in
        #: seconds since the epoch) and which attempt it was: a failed task's landing joins
        #: a regression check's batch too, and the planner hears how long the batch waited.
        self._landings: dict[str, tuple[float, int]] = {
            str(k): (float(v[0]), int(v[1])) for k, v in (saved.get("landings") or {}).items()
        }
        #: Per task that failed with a receipt whose commits landed, the attempt that
        #: landed them: its dependents' ``depends_on`` on it is met (:func:`graph.ready`).
        self._receipts_landed: dict[str, int] = {
            str(k): int(v) for k, v in (saved.get("receipts_landed") or {}).items()
        }
        #: An event that needs the planner's judgement waits: it wakes the planner
        #: at once. What a dead process left unread is read first.
        self._urgent = bool(self.events) or self.unread_reports
        self.revisions_since_progress = 0
        #: The planner's turn in flight, and the tasks the orchestrator added while it ran.
        self._planner: Future | None = None
        self._added_in_turn: set[str] = set()
        #: The turn in flight: how many of the events it was given, which attempt
        #: it is, and whether a worker reported while it ran.
        self._turn_events = 0
        self._turn_attempt = 0
        self._reports_in_turn = False
        #: The planner said DONE; it is told otherwise once nothing is left to run or check.
        self._done_said = False
        #: What the planner's last answer escalated: the work that does not wait on the
        #: operator's answer goes on, and the run ends ``escalated`` once nothing else moves.
        self._escalated = ""
        #: Why local variants last kept a holding contract from delivery, as the planner heard it.
        self._unpromoted_said = ""
        #: The delivery files that last kept a holding contract from full support, per
        #: marker, as the planner heard them.
        self._files_said: tuple[str, tuple[str, ...]] | None = None
        #: A running campaign was told to wrap up: full support holds.
        self._campaign_stopped = False
        #: The target's marker and the landing the last re-checks beside running work
        #: were chosen on.
        self._rechecked_at: tuple[str, str] | None = None
        #: Delivery-tier checks rehearsed in this process.
        self._rehearsed: set[str] = set()
        #: The task threads in flight, by task id, as the loop drives them.
        self._running: dict[str, Future] = {}
        #: Tasks whose work finished but some of whose checks raised: why, until recorded.
        self._inconclusive: dict[str, str] = {}
        #: Planner turns that raised in a row, and when the planner may be asked again.
        self._planner_failures = 0
        self._planner_retry_at = 0.0
        #: Running tasks whose thread needs no agent now: waiting on soft dependencies
        #: or running its checks. They take no task slot and hold no devices meanwhile.
        self._agentless: set[str] = set()
        #: Tasks whose commits were integrated on the target, and the attempt that did:
        #: a soft dependency is met then, without waiting for its checks.
        self._integrated: dict[str, int] = {}
        #: The foreground work that wants an agent now (ready or running required tasks
        #: that are not campaigns), as the loop last counted it.
        self._foreground_demand = 0
        #: When each ready task was first seen ready, for the order they start in.
        self._ready_since: dict[str, float] = {}
        #: Re-checks of an idle run in flight: what the planner is told if one fails, and the checks.
        self._settling: tuple[str, list[tuple[str, Future]]] | None = None
        #: The task whose change landed on the target last, in this process, and the
        #: target's marker just after it.
        self._landed: graph_mod.Task | None = None
        self._landed_marker = ""
        self._check_pool: ThreadPoolExecutor | None = None
        #: Checks passing on the target when progress was last measured.
        self._passed_seen = len(self._evaluated()[1].passed)
        #: Where correctness was last delivered (``correctness.jsonl``), and the failures on
        #: a later state of the target last journaled against it.
        self.milestone = contract_mod.latest_milestone(workspace)
        self._regressed: tuple[str, tuple[str, ...]] | None = None
        #: Notified whenever a task's outcome is recorded or the graph is revised, for
        #: the task threads waiting on soft dependencies; ``_closing`` once the loop ends.
        self._ended = threading.Condition()
        self._closing = False
        #: Done once the run is asked to stop (:func:`stoppable`); its result says why.
        with _STOPS_LOCK:
            self._stop: Future = _STOPS.get(self.workspace.resolve()) or Future()

    def _persist(self, what: str, write: Callable[[], None]) -> None:
        """``write`` now, or, when the filesystem is full, again until it lands."""
        try:
            write()
        except OSError as exc:
            if exc.errno not in STORAGE_FULL:
                raise
            if not self._unwritten:
                self.s.log(f"[storage] {what} could not be written ({exc}): nothing new starts")
            self._unwritten[what] = write
        else:
            self._unwritten.pop(what, None)

    def _written(self) -> bool:
        """Whether every state write a full filesystem refused has landed, trying each again."""
        if not self._unwritten:
            return True
        for what, write in list(self._unwritten.items()):
            self._persist(what, write)
        if not self._unwritten:
            self.s.log("[storage] the run's state is written again: work starts again")
        return not self._unwritten

    def _save_events(self) -> None:
        self._persist("events", self._write_events)

    def _write_events(self) -> None:
        state = {
            "events": [event.text for event in self.events],
            "unread_reports": self.unread_reports,
            "retries": self.retries,
            "not_started_attempts": self.not_started_attempts,
            "landings": {k: list(v) for k, v in self._landings.items()},
            "receipts_landed": self._receipts_landed,
        }
        write_text_atomic(self.events_path, json.dumps(state, indent=1))

    def _save_graph(self) -> None:
        self._persist(self.graph_path.name, self._write_graph)

    def _write_graph(self) -> None:
        """Write ``graph.yaml`` whole (:meth:`~.graph.Graph.save`): a planner turn may be reading it."""
        self.graph.save(self.graph_path)

    def _note(
        self,
        event: str,
        *,
        urgent: bool = False,
        outcome: _Outcome | None = None,
        report: bool = False,
        quiet: bool = False,
        added: Sequence[str] = (),
    ) -> None:
        """An event for the planner's next turn; an urgent one wakes it at once.

        ``outcome`` is the task run whose end it reports, for a revision that
        answers it before the planner reads it (:meth:`_drop_answered`).
        ``report`` marks a worker's report, which keeps the run from being
        delivered until the planner reads it (:attr:`unread_reports`). A
        ``quiet`` one wakes no turn: it waits for the next turn something else
        wakes. ``added`` are the tasks the orchestrator added that the event
        reports, for a revision that already holds them (:meth:`_drop_answered`).
        An event with neither ``outcome`` nor ``added`` is answered only by a
        turn that reads it.
        """
        self.events.append(Event(event, urgent, outcome, report, quiet, tuple(added)))
        if report:
            self.unread_reports = self._reports_in_turn = True
        if self._first_event_at is None and not quiet:
            self._first_event_at = time.monotonic()
        self._urgent = self._urgent or urgent

    # ----------------------------------------------------------------- checks
    def _check_by_key(self, key: str) -> contract_mod.Check | None:
        return next((c for c in self.contract.checks if c.key == key), None)

    def _delivery_keys(self) -> set[str]:
        return {c.key for c in self.contract.by_tier(contract_mod.TIER_DELIVERY)}

    def _run_check(self, check: contract_mod.Check) -> bool:
        """One check, in a pool thread, recorded under the target as it was when it started.

        A check the contract already counts as passed on this very target (its
        own verdict, or an equivalent check's) does not run again; that is
        asked as the check starts, and the target is read then.
        """
        marker = self.s.marker()
        if marker:
            with self._status_lock:
                evaluation = contract_mod.evaluate(
                    self.contract, self.status, marker=marker, holds=self.s.holds
                )
            if check.key in {c.key for c in evaluation.passed}:
                return True
        result = self.s.check(check)
        reads = self.s.narrow(result) if self.s.narrow is not None else ("", "")
        with self._status_lock:
            self.status.record(check.unit, result, marker=marker, inputs=check.inputs, reads=reads)
        return bool(result.passed)

    def _submit_checks(
        self, pool: ThreadPoolExecutor, keys: Iterable[str]
    ) -> list[tuple[str, Future]]:
        submitted = []
        for key in keys:
            check = self._check_by_key(key)
            if check is not None:
                submitted.append((key, pool.submit(self._run_check, check)))
        return submitted

    def _run_checks(self, keys: Iterable[str]) -> list[str]:
        """Run the named checks at once, record their verdicts, and return the keys that failed.

        Safe to call from several task threads at once: their checks share one
        bounded pool. Each verdict is recorded under the target as it was when
        its check started: a check that overlapped another task's integration
        is stale, not wrong.
        """
        keys = list(keys)
        if self._check_pool is None:
            # Outside :meth:`run`, the checks get a pool of their own.
            with ThreadPoolExecutor(max_workers=self.max_parallel_checks) as pool:
                return _failed(self._submit_checks(pool, keys))
        return _failed(self._submit_checks(self._check_pool, keys))

    def _run_task(self, task: graph_mod.Task) -> tuple[str, list[str]]:
        """A task's worker, then, in the same thread, the checks it claims.

        The delivery-tier checks it claims wait for the delivery tier, which
        runs once on the target as it is delivered.
        """
        try:
            with turn_priority(self._priority_of(task)):
                decision = self.s.work(task)
        except Exception as exc:  # a turn that raised is a failed task, not a crashed run
            return f"raised {type(exc).__name__}: {exc}", []
        if decision.partition("\n")[0] not in FINISHED:
            return decision, []
        delivery = self._delivery_keys()
        self._agentless.add(task.id)
        try:
            return decision, self._run_checks(k for k in task.satisfies if k not in delivery)
        except ChecksRaised as exc:
            # The work stands and has landed; only these checks' outcome is unknown.
            # They are missing on the target, so they run again as re-checks.
            self._inconclusive[task.id] = str(exc)
            return decision, exc.failed
        finally:
            self._agentless.discard(task.id)

    def _priority_of(self, task: graph_mod.Task) -> Priority:
        """The priority ``task``'s turns take agent slots at (``slots.Priority``).

        A campaign's are background; a required task on the longest open
        critical path is critical; other required tasks are required, and a
        ``nice_to_have`` one is side work.
        """
        if task.kind in BACKGROUND_KINDS:
            return Priority.BACKGROUND
        if task.priority != "required":
            return Priority.SIDE
        path = graph_mod.critical_path_minutes(self.graph)
        open_paths = [
            path.get(t.id, 0)
            for t in self.graph.tasks
            if t.status in ("pending", "running")
            and t.priority == "required"
            and t.kind not in BACKGROUND_KINDS
        ]
        longest = max(open_paths, default=0)
        return Priority.CRITICAL if path.get(task.id, 0) >= longest else Priority.REQUIRED

    def idle_agent_slots(self) -> int:
        """Agent slots the open foreground work leaves idle: what a campaign's turns may use."""
        return max(0, AgentGate(self.max_parallel).task_slots - self._foreground_demand)

    def agentless(self, task_id: str) -> bool:
        """Whether running ``task_id`` needs no agent and no device now (waiting or checking)."""
        return task_id in self._agentless

    def mark_integrated(self, task_id: str, attempt: int) -> None:
        """``task_id``'s commits are on the target: the tasks that soft-depend on it may go on."""
        with self._ended:
            self._integrated[task_id] = attempt
            task = self.graph.by_id().get(task_id)
            if task is not None:
                # What landed last: stale checks run again beside the work once per landing.
                self._landed, self._landed_marker = task, self.s.marker()
            self._ended.notify_all()

    def _evaluated(self) -> tuple[str, contract_mod.Evaluation]:
        """The target's marker now, and the contract evaluated on it."""
        marker = self.s.marker()
        with self._status_lock:
            return marker, contract_mod.evaluate(
                self.contract, self.status, marker=marker, holds=self.s.holds
            )

    def _completion(
        self,
        marker: str,
        evaluation: contract_mod.Evaluation,
        files: Sequence[str] | None = None,
        *,
        unpromoted: bool | None = None,
    ) -> Completion:
        """Fresh disk evidence and scheduler state, assessed by the shared completion policy."""
        missing = (
            list(files)
            if files is not None
            else (
                list(self.s.delivery_files()) if evaluation.holds and self.s.delivery_files else []
            )
        )
        result = assess(
            evaluation,
            files=missing,
            unreviewed=self._unreviewed_landings(),
            unpromoted=(
                bool(self.s.promotion and self.s.promotion(self.graph).entries)
                if unpromoted is None
                else unpromoted
            ),
            required_open=self._required_open(campaign=False),
            unread_reports=self.unread_reports,
        )
        said = (marker, tuple(missing))
        if evaluation.holds and missing and said != self._files_said:
            self._files_said = said
            self._note(
                "every required check holds, but the delivery files on disk do not check out:\n"
                + "\n".join(f"- {m}" for m in missing)
                + "\nadd or fix the task that writes them; the run is not delivered, and correctness "
                "is not recorded, before they do",
                urgent=True,
            )
            self._save_events()
        return result

    def completion(self, files: Sequence[str]) -> Completion:
        """Final delivery re-evaluates current verdicts, graph, promotions and regression coverage."""
        return self._completion(*self._evaluated(), files=files)

    def _correctness(
        self, marker: str, evaluation: contract_mod.Evaluation, completion: Completion | None = None
    ) -> None:
        """Record correctness delivered on a state of the target that is fully supported.

        The first state on which full support holds (:class:`Completion`)
        becomes the milestone: the workflow names its commit and appends it to
        ``correctness.jsonl`` (:attr:`Services.correct`), with no planner turn.
        The next state full support holds on moves it forward. Required checks that fail on a later state are journaled
        against it, once per state and set of failures.
        """
        if not marker or not evaluation.passed:
            return
        if self.s.bringup is not None:
            self.s.bringup(marker, evaluation)
        if self.s.cells is not None:
            self.s.cells(marker, evaluation)
        if self.s.correct is None:
            return
        held = self.milestone
        if evaluation.holds:
            if held is not None and held.marker == marker:
                return
            if not (completion or self._completion(marker, evaluation)).supported:
                return
            self.milestone = self.s.correct(marker, evaluation) or held
            return
        failed = tuple(c.key for c, _ in evaluation.failed)
        if held is None or not failed or marker == held.marker:
            return
        if self._regressed == (marker, failed):
            return
        self._regressed = (marker, failed)
        self.s.log(
            f"correctness stays delivered at {held.sha[:10]}: on the target as it is now "
            f"{', '.join(failed)} failed"
        )

    def _gated(self) -> dict[str, list[str]]:
        """Per gated kind, the checks it is still waiting for.

        A check lets a gated task start once its latest verdict passed, on
        whatever state of the target it measured: a ``tune`` task measures the
        commit it pins when it starts, so an integration that leaves the pass
        stale does not hold it back again. A later failure does.
        """
        if not self.start_after:
            return {}
        with self._status_lock:
            latest = self.status.latest()
        passed = {key for key, verdict in latest.items() if verdict.passed}
        waiting = {
            kind: [k for k in keys if k not in passed] for kind, keys in self.start_after.items()
        }
        return {kind: keys for kind, keys in waiting.items() if keys}

    def _required_open(self, *, campaign: bool = True) -> bool:
        """Whether a required task is pending or running.

        With ``campaign=False`` a campaign, and a task that waits for one
        (:meth:`_waiting_on_campaign`), does not count.
        """
        beside = set() if campaign else self._waiting_on_campaign()
        return any(
            t.priority == "required" and t.status in ("pending", "running") and t.id not in beside
            for t in self.graph.tasks
        )

    def _waiting_on_campaign(self) -> set[str]:
        """The open campaigns (:data:`BACKGROUND_KINDS`) and the open tasks that depend on one.

        Directly or through other open tasks: such a task cannot start before
        the campaign ends, which is hours away.
        """
        tasks = self.graph.by_id()
        waits: dict[str, bool] = {}

        def waiting(task_id: str) -> bool:
            if task_id not in waits:
                waits[task_id] = False  # validated acyclic; a cycle that slipped in ends here
                task = tasks.get(task_id)
                waits[task_id] = (
                    task is not None
                    and task.status in ("pending", "running")
                    and (task.kind in BACKGROUND_KINDS or any(map(waiting, task.depends_on)))
                )
            return waits[task_id]

        return {task_id for task_id in tasks if waiting(task_id)}

    def _foreground(self, running: Mapping[str, Future]) -> list[str]:
        """The running tasks that are not campaigns, which run for hours beside the graph."""
        kinds = {t.id: t.kind for t in self.graph.tasks}
        return [task_id for task_id in running if kinds.get(task_id) not in BACKGROUND_KINDS]

    def _stop_campaigns(self, running: Mapping[str, Future]) -> None:
        """Full support holds: tell the running campaign to wrap up, once.

        It dispatches nothing new and ends with its best kept tree, which is
        reported as an artifact of the run; the run is delivered once it has.
        """
        if self._campaign_stopped:
            return
        kinds = {t.id: t.kind for t in self.graph.tasks}
        campaigns = [tid for tid in running if kinds.get(tid) in BACKGROUND_KINDS]
        if not campaigns:
            return
        self._campaign_stopped = True
        reason = "full support holds on the target as it is"
        self.s.log(
            f"[support] {reason}: {', '.join(campaigns)} wraps up, then the run is delivered"
        )
        self.s.stop_campaign(reason)

    def _settle(self, busy: bool = False) -> None:
        """Keep the target verified as it changes; then, with no required task open, the delivery tier.

        ``busy``, with foreground work running: each time a change lands, the
        first :data:`RECHECKS_WHILE_BUSY` stale checks in the contract's order
        run again beside it, and the delivery tier of a member whose other
        checks all hold is rehearsed once (:meth:`_rehearsal`). Otherwise every
        stale check runs again; and once every other required check holds and
        no required task is open but a campaign, the delivery tier runs on the
        target as it is. Both run on the check pool; :meth:`_settled` reads the
        verdicts once they are all in, and tells the planner what failed.

        A stale accuracy check runs again at a milestone only: once no required
        task is open, or for the bring-up before it was named. Until then the
        cheaper checks of each landing (the logit gate, the boot's greedy
        tokens) stand watch.
        """
        claimed = {k for t in self.graph.tasks if t.status == "done" for k in t.satisfies}
        delivery = self._delivery_keys()
        marker, evaluation = self._evaluated()
        stale = [c.key for c in evaluation.missing if c.key in claimed and c.key not in delivery]
        if self._required_open(campaign=False):
            bringup = (
                {c.key for c in contract_mod.bringup_checks(self.contract)}
                if contract_mod.bringup_milestone(self.workspace) is None
                else set()
            )
            stale = [k for k in stale if k not in self._accuracy_keys or k in bringup]
        if busy or (not stale and self._required_open(campaign=False)):
            keys: list[str] = []
            # Once per landing the loop knows made the target what it is.
            landed = (marker, self._landed.id if self._landed is not None else "")
            known = self._landed is not None and self._landed_marker == marker
            if stale and known and landed != self._rechecked_at:
                self._rechecked_at = landed
                keys = stale[:RECHECKS_WHILE_BUSY]
            keys += self._rehearsal(evaluation)
            what = "checks run beside the work in flight failed on the current target: "
        else:
            what, keys = (
                "checks that held before failed on the current target: ",
                stale,
            )
            if not stale:
                # A check that already failed on this very target stays failed until
                # the target changes; running it again would measure the same thing.
                missing = {c.key for c in evaluation.missing}
                failed_here = {c.key for c, _ in evaluation.failed}
                if not missing or not (missing | failed_here) <= delivery:
                    return
                what, keys = "delivery checks failed: ", sorted(missing)
        submitted = self._submit_checks(self._check_pool, keys) if keys else []
        if submitted:
            self._settling = (what, submitted)

    def _rehearsal(self, evaluation: contract_mod.Evaluation) -> list[str]:
        """Delivery-tier checks to rehearse now, each once: those whose member is otherwise green.

        A member whose every other required check passed on the target as it
        is gets its delivery-tier checks run once, ahead of the final tree,
        as advisory: a failure reaches the planner, and the final tree runs
        them again whatever they found.
        """
        passed = {c.key for c in evaluation.passed}
        required = self.contract.required()
        due = []
        for check in self.contract.by_tier(contract_mod.TIER_DELIVERY):
            if check.key in self._rehearsed or check.key in passed:
                continue
            before = [
                c
                for c in required
                if c.member == check.member and c.tier != contract_mod.TIER_DELIVERY
            ]
            if before and all(c.key in passed for c in before):
                self._rehearsed.add(check.key)
                due.append(check.key)
        return due

    def _settled(self) -> None:
        what, submitted = self._settling
        self._settling = None
        try:
            failed = _failed(submitted)
        except ChecksRaised as exc:
            # No verdict was recorded for these: they are still due and run again.
            failed = exc.failed
            self.s.log(f"[settle] re-checks raised, to run again: {exc}")
        if failed:
            self._note(what + ", ".join(failed), urgent=True)
            self._save_events()
        self._correctness(*self._evaluated())

    # ---------------------------------------------------------------- planner
    def _wake_planner(self, pool: ThreadPoolExecutor, refusals: Sequence[str] = ()) -> None:
        """Start a planner turn on the events so far; what arrives meanwhile waits for the next."""
        self._turn_attempt = self._turn_attempt + 1 if refusals else 1
        self._turn_events = len(self.events)
        self._reports_in_turn = False
        self._urgent = False
        self._first_event_at = None
        self._added_in_turn = set()
        marker, evaluation = self._evaluated()
        turn = PlannerTurn(
            evaluation,
            [event.text for event in self.events],
            list(refusals),
            marker=marker,
            regression=self._regression_standing(),
        )
        self._planner = pool.submit(run_at, Priority.PLANNER, self.s.plan, turn)

    def _planner_returned(self, pool: ThreadPoolExecutor) -> str | None:
        """Apply what the planner's thread returned; a final answer is the run's verdict.

        A refused graph goes back at once, up to :data:`MAX_REFUSALS` turns.
        An admissible graph is adopted at once, and the regression checks its
        changes to shared code need are added to it (:meth:`_follow_shared_changes`).
        The tasks the orchestrator added while the turn ran, which its graph
        could not name, are kept (:meth:`_added_during`). An adopted revision
        answers the events of its turn that it re-planned (:meth:`_drop_answered`).
        """
        future, self._planner = self._planner, None
        try:
            answer = future.result()
        except Exception as exc:  # a turn that raised costs a retry, not the run
            return self._planner_raised(exc)
        self._planner_failures = 0
        self._done_said = answer.decision == DONE
        self._escalated = answer.summary if answer.decision == ESCALATE else ""
        if answer.decision in (ESCALATE, CEILING, DONE):
            self._consume_turn_events()
        if answer.decision == ESCALATE:
            self.s.log(
                f"[planner] escalated: {answer.summary}. The work that does not wait on the "
                "answer goes on; answer with `trtllm-modeling-v2 feedback`."
            )
            self._note(
                f"you escalated ({answer.summary}); the work that does not wait on the "
                "operator's answer goes on",
                quiet=True,
            )
            return None
        if answer.decision == CEILING:
            self.s.log(f"[planner] ceiling: {answer.summary}")
            return "ceiling"
        if answer.decision == DONE:
            return None
        if answer.graph is None:
            refusals = [answer.problem or "GRAPH_UPDATED without a graph"]
        else:
            # As the planner wrote them, before the orchestrator's own are carried over.
            written = {t.id: (t.status, t.attempts) for t in answer.graph.tasks}
            proposed = graph_mod.Graph(
                tasks=[*answer.graph.tasks, *self._added_during(answer.graph)],
                revisions=answer.graph.revisions,
            )
            adopted, refusals = graph_mod.apply_revision(
                self.graph,
                proposed,
                answer.summary,
                not_started_attempts=self.not_started_attempts,
                **self._admission(),
            )
            if not refusals:
                self._consume_turn_events()
                self.graph = adopted
                self._added_in_turn = set()
                self._drop_answered(written)
                self._save_graph()
                self._wake_soft_waiters()
                self.revisions_since_progress += 1
                self._persist("state", self.s.checkpoint)
                return None
            self.s.log("[planner] graph refused: " + "; ".join(refusals))
            with contextlib.suppress(OSError):  # a line of metrics, not state
                metrics_mod.record_refusal(self.workspace, list(refusals))
        if self._turn_attempt >= MAX_REFUSALS:
            return "planner-refused"
        # The events stay unread: the next turn reads them again, with why it was refused.
        self._wake_planner(pool, refusals)
        return None

    def _consume_turn_events(self) -> None:
        """The turn's answer was taken: the events it was given are read, those since are not."""
        del self.events[: self._turn_events]
        self._turn_events = 0
        self.unread_reports = self._reports_in_turn
        if self._loud():
            # What arrived during the turn waits a window of its own.
            self._first_event_at = time.monotonic()
        self._save_events()

    def _planner_raised(self, exc: Exception) -> str | None:
        """A planner turn raised (it timed out, crashed, or wrote nothing readable).

        Its events stay unread, and the planner is asked again after a wait
        that doubles with each failure in a row; the run goes on meanwhile.
        After :data:`MAX_PLANNER_FAILURES` in a row the run stops with
        ``planner-failed``.
        """
        self._planner_failures += 1
        what = f"{type(exc).__name__}: {exc}"
        self.s.log(f"[planner] the turn raised ({self._planner_failures} in a row): {what}")
        if self._planner_failures >= MAX_PLANNER_FAILURES:
            return f"planner-failed: {what}"
        wait_s = PLANNER_RETRY_S * 2 ** (self._planner_failures - 1)
        self._planner_retry_at = time.monotonic() + wait_s
        self._note(
            f"your previous turn raised {what}; this turn reads the same events again",
            urgent=True,
        )
        self._save_events()
        return None

    def _added_during(self, proposal: graph_mod.Graph) -> list[graph_mod.Task]:
        """The tasks the orchestrator added while the turn ran that ``proposal`` does not name.

        The turn read the graph before they existed, so leaving them out is
        not dropping them; they are kept as they are, and the next turn may
        reshape them.
        """
        named = {t.id for t in proposal.tasks}
        return [
            copy.deepcopy(t)
            for t in self.graph.tasks
            if t.id in self._added_in_turn and t.id not in named
        ]

    def _drop_answered(self, written: Mapping[str, tuple[str, int]]) -> None:
        """Drop the events that arrived during the turn which the revision just adopted answered.

        An event that reports how a task's run ended is answered when the
        revision re-planned that task past it (:func:`_answered`); one that
        reports tasks the orchestrator added, when the revision holds them all;
        every other event, a finished run's report among them, is kept.
        ``written`` is each task's status and attempt as the planner wrote
        them, before the orchestrator's own were carried over.
        """
        tasks = self.graph.by_id()
        kept, dropped = [], []
        for event in self.events:
            answered = (event.outcome is not None and _answered(event.outcome, written, tasks)) or (
                bool(event.added) and all(task_id in written for task_id in event.added)
            )
            (dropped if answered else kept).append(event)
        if not dropped:
            return
        self.events = kept
        if any(event.report for event in dropped) and not any(e.report for e in self.events):
            self.unread_reports = False
        self._urgent = any(event.urgent for event in self.events)
        if not self._loud():
            self._first_event_at = None
        self._save_events()
        self.s.log(
            f"[planner] revision {len(self.graph.revisions)} already answered "
            f"{len(dropped)} event(s) of its turn: " + "; ".join(event.text for event in dropped)
        )

    def _only_background(self, running: dict) -> bool:
        """Whether every running task is one that runs for hours beside the graph (a campaign)."""
        if not running or self._settling is not None:
            return False
        return not self._foreground(running)

    def _loud(self) -> bool:
        """Whether an event waits that may wake a turn: any but a quiet one."""
        return any(not event.quiet for event in self.events)

    def _answered_ahead(self, task: graph_mod.Task) -> bool:
        """Whether the graph already answered ``task`` finishing green: it needs no turn of its own.

        It satisfies no contract check, so no verdict and no runtime fact
        moved, and an open task in the graph depends on it: that one starts on
        it with its receipt in hand, whether or not a turn reads it first.
        """
        return not task.satisfies and any(
            task.id in (*t.depends_on, *t.depends_on_soft)
            for t in self.graph.tasks
            if t.id != task.id and t.status in ("pending", "running")
        )

    def _events_due(self) -> bool:
        """Events wait for the window while work is in flight, unless one needs judgement now.

        Quiet events alone are never due: they wait for the next turn
        something else wakes, or for the graph to run out of work.
        """
        if not self._loud():
            return False
        if self._urgent or self._first_event_at is None:
            return True
        return time.monotonic() - self._first_event_at >= self.event_window_s

    # ------------------------------------------------------------------- loop
    def _finish(self, task: graph_mod.Task, decision: str, failed: list[str]) -> None:
        decision, _, note = decision.partition("\n")
        note = note.strip()
        blocks_work = task.priority != "nice_to_have" and any(
            task.id in other.depends_on and other.status == "pending" for other in self.graph.tasks
        )
        # Counted per attempt: a retry the planner asks for gets its own.
        run_key = f"{task.id}@{task.attempts}"
        retried = self.retries.get(run_key, 0)
        if decision not in FINISHED and _not_the_task(decision) and retried < INFRA_RETRIES:
            # Not the task's doing: it starts again as it is, and the planner hears why.
            self.retries[run_key] = retried + 1
            task.status = "pending"
            event = (
                f"task {task.id} ({task.kind}) ended {decision}, a failure of the run's "
                f"infrastructure rather than of the task; it starts again (retry "
                f"{retried + 1} of {INFRA_RETRIES})" + (f": {note}" if note else "")
            )
            self.s.log(f"[retry] {event}")
            self._note(event)
        else:
            ended = "done" if decision in FINISHED and not failed else "failed"
            if decision.startswith(f"raised {TaskNotStarted.__name__}:"):
                unstarted = self.not_started_attempts.setdefault(task.id, [])
                if task.attempts not in unstarted:
                    unstarted.append(task.attempts)
            if ended == "failed" and task.priority == "nice_to_have":
                ended = "abandoned"
            outcome = _Outcome(task.id, task.attempts, ended)
            if note and task.kind in BACKGROUND_KINDS:
                # A campaign's report names its best branch, an artifact of the run that
                # no task acts on: journaled and read by the next turn, waiting for none.
                self.s.log(f"[{task.kind}] {task.id} reports: {note}")
                self._note(f"task {task.id} ({task.kind}) reports: {note}", quiet=True)
            elif note:
                # A run that did not finish reports part of how it ended (what became
                # of a failed receipt's commits): a revision that re-planned the task
                # past the run answers both. A finished run's report may still need
                # acting on, so nothing answers it but a turn.
                self._note(
                    f"task {task.id} ({task.kind}) reports: {note}",
                    urgent=ended == "failed" and blocks_work,
                    outcome=None if decision in FINISHED else outcome,
                    report=True,
                )
            if decision not in FINISHED:
                task.status = "failed"
                said = f"task {task.id} ({task.kind}) ended {decision or 'with no decision'}"
                if task.kind in INTEGRATED_KINDS and self.s.landed(task):
                    self._landings[task.id] = (time.time(), task.attempts)
                    self._receipts_landed[task.id] = task.attempts
                    said += self._landed_receipt_note(task)
                self._note(said, urgent=blocks_work, outcome=outcome)
            else:
                task.status = "failed" if failed else "done"
                if task.kind in INTEGRATED_KINDS:
                    # Integrated before its checks ran, whatever they found.
                    self._landed, self._landed_marker = task, self.s.marker()
                    self._landings[task.id] = (time.time(), task.attempts)
                unknown = self._inconclusive.pop(task.id, "")
                self._note(
                    _check_task_event(task, failed)
                    if task.kind == graph_mod.CHECK_KIND
                    else f"task {task.id} ({task.kind}) {decision}; "
                    + (f"checks failed: {', '.join(failed)}" if failed else "its checks passed")
                    + (
                        f"; some checks could not run ({unknown}): they run again as re-checks"
                        if unknown
                        else ""
                    ),
                    urgent=bool(failed) and blocks_work,
                    outcome=outcome,
                    quiet=not failed
                    and not unknown
                    and (task.kind == graph_mod.CHECK_KIND or self._answered_ahead(task)),
                )
            if task.status == "failed" and task.priority == "nice_to_have":
                task.status = "abandoned"
                self._note(f"nice_to_have task {task.id} abandoned", outcome=outcome)
            if task.status in ("failed", "abandoned"):
                self._note_soft_dependents(task)
            if task.kind == "survey" and task.status == "done":
                self._add_kernel_tasks(task)
            self._promote()
        # Progress is a task done, a change landed on the target (a failed
        # receipt's included) or a check turning green. The task's own checks
        # already ran in its thread, so they count against what was passing the
        # last time this was measured, not against now.
        marker, evaluation = self._evaluated()
        passed = len(evaluation.passed)
        landed = task.id in self._landings and self._landings[task.id][1] == task.attempts
        if task.status == "done" or landed or passed > self._passed_seen:
            self.revisions_since_progress = 0
        self._passed_seen = passed
        self._save_graph()
        self._save_events()
        self._correctness(marker, evaluation)
        self._persist("state", self.s.checkpoint)
        self.s.finished(task)
        self._wake_soft_waiters()

    def _wire_core_rows(self, rows: list[graph_mod.Task]) -> str:
        """Make each pending core assembly soft-depend on ``rows``; the ids of those wired.

        An assembly that already names a row keeps it as it is; one the edges
        would put on a cycle is left unwired.
        """
        wired = []
        for task in self.graph.tasks:
            if not (
                task.kind == "assemble"
                and (task.ref or task.id) == vocab_mod.UNIT_CORE
                and task.status == "pending"
            ):
                continue
            new = [
                r.id
                for r in rows
                if r.id not in task.depends_on and r.id not in task.depends_on_soft
            ]
            if not new:
                continue
            task.depends_on_soft.extend(new)
            if graph_mod._cycle(self.graph.by_id()):
                del task.depends_on_soft[-len(new) :]
                continue
            wired.append(task.id)
        return ", ".join(wired)

    def _landed_receipt_note(self, task: graph_mod.Task) -> str:
        """What the planner hears of the tasks that hard-depend on ``task``, whose receipt landed."""
        dependents = [
            t.id for t in self.graph.tasks if task.id in t.depends_on and t.status == "pending"
        ]
        if not dependents:
            return ""
        return (
            f"; its receipt's commits are on the target, so {', '.join(dependents)}, which "
            f"depend on it, start without waiting for a revision: revise them if they need more "
            f"than it landed"
        )

    def _note_soft_dependents(self, task: graph_mod.Task) -> None:
        """Tell the planner which open tasks soft-depend on ``task``, which ended without landing.

        They go on: one running integrates without its change, one pending starts
        without it, and each says so in its receipt. The planner may revise them.
        """
        dependents = [
            t.id
            for t in self.graph.tasks
            if task.id in t.depends_on_soft and t.status in ("pending", "running")
        ]
        if dependents:
            self._note(
                f"{', '.join(dependents)} soft-depend on {task.id}, which ended {task.status}: "
                f"they go on without its change, and say so in their receipts; revise them if "
                f"they cannot",
                outcome=_Outcome(task.id, task.attempts, task.status),
            )

    def _wake_soft_waiters(self) -> None:
        with self._ended:
            self._ended.notify_all()

    def soft_dependencies(self, task_id: str) -> dict[str, str] | None:
        """Wait until every soft dependency of ``task_id`` has landed or ended; how each did.

        For a task's thread, before its integration: the dependencies the
        graph names for it now, each ``integrated`` once its commits are on
        the target (:meth:`mark_integrated`; its checks need not have run),
        else ``done``, ``failed`` or ``abandoned``; one no longer in the graph
        holds nothing back and is left out. ``None`` when the loop ended
        first: no outcome of the task will be read, so it lands nothing.
        While it waits the task needs no agent and holds no device
        (:meth:`agentless`), and the dependencies it waits for start first.
        """
        self._agentless.add(task_id)
        try:
            with self._ended:
                while not self._closing:
                    ended = self._soft_standing(task_id)
                    if all(status not in ("pending", "running") for status in ended.values()):
                        return ended
                    self._ended.wait(SOFT_POLL_S)
            return None
        finally:
            self._agentless.discard(task_id)

    def _soft_standing(self, task_id: str) -> dict[str, str]:
        """Each soft dependency of ``task_id``: ``integrated``, or its status."""
        tasks = self.graph.by_id()
        task = tasks.get(task_id)
        soft = task.depends_on_soft if task is not None else []
        return {
            dep: (
                "integrated"
                if tasks[dep].status in ("pending", "running")
                and self._integrated.get(dep) == tasks[dep].attempts
                else tasks[dep].status
            )
            for dep in soft
            if dep in tasks
        }

    def _closing_holds(self) -> set[str]:
        """With no required task open, the pending ``nice_to_have`` tasks that would change the tree.

        The run is closing: what is left is checking the target as it is, and
        a landing now would make every check stale again. They wait for a
        planner turn to require them or for the run to end.
        """
        if self._required_open(campaign=False):
            return set()
        return {
            t.id
            for t in self.graph.tasks
            if t.status == "pending" and t.priority != "required" and t.kind in INTEGRATED_KINDS
        }

    def _waited_for(self) -> set[str]:
        """Pending tasks a running task waits for at its integration: they start first."""
        tasks = self.graph.by_id()
        return {
            dep
            for tid in list(self._agentless)
            if tid in tasks
            for dep, status in self._soft_standing(tid).items()
            if status == "pending"
        }

    def _held(self, task: graph_mod.Task) -> int:
        """What ``task`` holds of the free devices while it runs: none for a ``check`` task."""
        return 0 if task.kind == graph_mod.CHECK_KIND else self.s.held_gpus(task)

    def _start_check_task(self, task: graph_mod.Task) -> Future:
        """Submit a ``check`` task's checks to the check pool; the future of its outcome.

        No agent turn runs and no task slot is taken: its checks wait for
        devices on the check pool as every check does.
        The future holds what a worker's thread returns, :data:`CHECKED` and
        the keys that failed, once every check is in. Its delivery-tier checks
        wait for the delivery tier, as any task's do. A regression check runs
        the package tests instead (``Services.regression``), and what failed is
        named the same way.
        """
        if task.regression:
            outcome = Future()
            if self.s.regression is None:
                outcome.set_result((CHECKED, []))
                return outcome
            ran = self._check_pool.submit(self.s.regression, copy.deepcopy(task))

            def tests_in(done: Future) -> None:
                try:
                    outcome.set_result((CHECKED, list(done.result())))
                except Exception as exc:  # no verdict: the loop starts the task again, once
                    outcome.set_exception(exc)

            ran.add_done_callback(tests_in)
            return outcome
        delivery = self._delivery_keys()
        submitted = self._submit_checks(
            self._check_pool, [k for k in task.satisfies if k not in delivery]
        )
        outcome: Future = Future()
        if not submitted:
            outcome.set_result((CHECKED, []))
            return outcome
        left = [len(submitted)]
        lock = threading.Lock()

        def one_in(_: Future) -> None:
            with lock:
                left[0] -= 1
                if left[0]:
                    return
            try:
                failed = [key for key, future in submitted if not future.result()]
            except Exception as exc:  # no verdict: the loop starts the task again, once
                outcome.set_exception(exc)
                return
            outcome.set_result((CHECKED, failed))

        for _, future in submitted:
            future.add_done_callback(one_in)
        return outcome

    # ------------------------------------------------------------- regression
    def _regression_step(self) -> set[str]:
        """The ready regression checks to keep waiting; those with nothing to check close.

        Only while the run has shared code to check (its own paths are
        known). One regression check runs at a time: while one runs, the
        others wait and what lands meanwhile joins the next batch. With none
        running, a check whose batch is empty closes at once
        (:meth:`_close_reviewed`), and of the rest the one on the longest
        chain starts.
        """
        own = self.validation["own_paths"]
        if not own:
            return set()
        reviews = [
            t
            for t in graph_mod.ready(self.graph, self._receipts_landed)
            if graph_mod.is_regression_check(t)
        ]
        if not reviews:
            return set()
        held = {t.id for t in reviews}
        if any(
            t.status == "running" and graph_mod.is_regression_check(t) for t in self.graph.tasks
        ):
            return held
        batch = graph_mod.regression_batch(self.graph, own, self._landed_attempts())
        if not batch:
            for review in reviews:
                self._close_reviewed(review)
            return set()
        path = graph_mod.critical_path_minutes(self.graph)
        first = max(reviews, key=lambda t: path.get(t.id, 0))  # the first of equals, in graph order
        return held - {first.id}

    def _landed_attempts(self) -> dict[str, int]:
        """Per task whose commits landed, the attempt that landed them last."""
        return {tid: attempt for tid, (_, attempt) in self._landings.items()}

    def _record_batch(self, review: graph_mod.Task) -> None:
        """Record, as a regression check starts, the batch it covers (:attr:`~.graph.Task.covers`)."""
        own = self.validation["own_paths"]
        if not own or not graph_mod.is_regression_check(review):
            return
        tasks = self.graph.by_id()
        batch = graph_mod.regression_batch(self.graph, own, self._landed_attempts())
        review.covers = [graph_mod.landing(tasks[tid]) for tid in batch]
        self.s.log(
            f"[regression] {review.id} starts on the changes to shared code landed since the "
            f"last check: {', '.join(batch) or 'none'}"
        )

    def _close_reviewed(self, review: graph_mod.Task) -> None:
        """Close a regression check whose batch is empty: earlier ones covered everything.

        Every change to shared code it follows landed and was covered by a
        regression check that ran and passed; nothing runs for it.
        """
        tasks = self.graph.by_id()
        follows = {graph_mod.landing(tasks[d]) for d in review.depends_on if d in tasks}
        by = sorted(
            t.id
            for t in self.graph.tasks
            if graph_mod.is_regression_check(t)
            and t.status == "done"
            and set(t.covers or ()) & follows
        )
        review.status, review.covers = "done", []
        covered = f": {', '.join(by)} covered what it follows" if by else ""
        self._note(
            f"task {review.id} (regression check) closed: no change to shared code was left "
            f"for it to check{covered}",
            outcome=_Outcome(review.id, review.attempts, "done"),
        )
        self.revisions_since_progress = 0
        self._save_graph()
        self._save_events()
        self._persist("state", self.s.checkpoint)
        self.s.finished(review)
        self._wake_soft_waiters()

    def _regression_standing(self) -> str:
        """Where the regression checks stand, for the planner; "" without own paths."""
        own = self.validation["own_paths"]
        if not own:
            return ""
        batch = graph_mod.regression_batch(self.graph, own, self._landed_attempts())
        reviews = [t for t in self.graph.tasks if graph_mod.is_regression_check(t)]
        running = [t.id for t in reviews if t.status == "running"]
        waiting = [t.id for t in reviews if t.status == "pending"]
        said = []
        if batch:
            landed = [self._landings[tid][0] for tid in batch if tid in self._landings]
            minutes = (time.time() - min(landed)) / 60 if landed else None
            since = f", the first landed {minutes:.0f} min ago" if minutes is not None else ""
            said.append(
                f"Landed changes to shared code no regression check covers yet, the batch the "
                f"next one takes: {', '.join(batch)}{since}."
            )
        else:
            said.append("Every landed change to shared code is covered by a regression check.")
        if running:
            said.append(f"Running: {', '.join(running)}; the others wait for it.")
        if waiting:
            said.append(
                f"Not started: {', '.join(waiting)}. A regression check starts once its "
                f"dependencies are done and no other runs; one whose batch is empty closes at "
                f"once. The orchestrator adds and wires them: leave them as they are."
            )
        return " ".join(said)

    def _unreviewed_landings(self) -> list[str]:
        """The landed changes to shared code no regression check covers (none without own paths)."""
        own = self.validation["own_paths"]
        if not own:
            return []
        return graph_mod.regression_batch(self.graph, own, self._landed_attempts())

    def _add_kernel_tasks(self, survey: graph_mod.Task) -> None:
        """One kernel task per queue row of the vocabulary the survey wrote, with no planner turn.

        Each works on its row (``ref``) and touches the row's entry and its test
        (:func:`~..operators.coverage.entry_paths`). A row that depends on
        others waits for their tasks, and a row a kernel task already names is
        left to that task. Rows of the core unit are wired at once into the
        core assembly that has not started (:meth:`_wire_core_rows`), so they
        rank on its critical path.
        """
        try:
            queue = vocab_mod.load_vocabulary(self.workspace / vocab_mod.VOCABULARY_FILENAME)
        except vocab_mod.VocabularyError as exc:
            self.s.log(f"[survey] no kernel tasks added: {exc}")
            return
        own = self.validation["own_paths"]
        target = survey.ref or (own[0] if own else "")
        kernels = {t.ref: t.id for t in self.graph.tasks if t.kind == "kernel"}
        taken = {t.id for t in self.graph.tasks}
        units = {c.unit for c in self.contract.checks}
        required = {c.unit for c in self.contract.required()}
        rows = [item for item in queue if str(item.n) not in kernels]
        added: list[graph_mod.Task] = []
        unplaced: list[str] = []
        for item in rows:
            paths = vocab_mod.entry_paths(item, target, self.repo_root or None)
            if paths is None or not all(
                graph_mod.inside(p, self.validation["allowed_paths"]) for p in paths
            ):
                unplaced.append(str(item.n))
                continue
            task_id = _kernel_task_id(item, taken)
            taken.add(task_id)
            kernels[str(item.n)] = task_id
            added.append(
                graph_mod.Task(
                    id=task_id,
                    kind="kernel",
                    goal=(
                        f"{item.mode} `{item.target}` as vocabulary queue row {item.n} says, "
                        f"with its GPU test driving the cells it certifies"
                    ),
                    touches=list(paths),
                    # A verify row's kernel writer runs tests whose jobs take their own devices.
                    gpus=0 if item.mode == vocab_mod.MODE_VERIFY else 1,
                    est_minutes=120 if item.writes_implementation else 60,
                    priority=("nice_to_have" if item.unit in units - required else "required"),
                    ref=str(item.n),
                )
            )
        if not added and not unplaced:
            return
        by_row = {item.n: item for item in queue}
        for task in added:
            deps = by_row[int(task.ref)].depends_on
            task.depends_on = [kernels[str(n)] for n in deps if str(n) in kernels]
        names = ", ".join(t.id for t in added)
        if added:
            self.graph.tasks.extend(added)
            wired = self._wire_core_rows(
                [t for t in added if by_row[int(t.ref)].unit == vocab_mod.UNIT_CORE]
            )
            if wired:
                names += f" ({wired} soft-depends on the core rows)"
            self.graph.revisions.append(
                graph_mod.Revision(
                    len(self.graph.revisions) + 1,
                    f"orchestrator: one kernel task per queue row of the vocabulary {survey.id} "
                    f"wrote: {names}",
                )
            )
            if self._planner is not None:
                self._added_in_turn |= {t.id for t in added}
        shared = [t.id for t in added if graph_mod.shared_touches(t, own)] if own else []
        said = [f"task {survey.id} wrote {len(queue)} vocabulary queue row(s)"]
        if added:
            said.append(
                f"the orchestrator added a kernel task for {len(added)} of them, which start "
                f"now: {names}"
            )
        if unplaced:
            said.append(
                f"rows {', '.join(unplaced)} name an entry it could not place (under a target "
                f"it does not know, or outside the paths the run may change): give them tasks"
            )
        if shared:
            self._follow_shared_changes()
            said.append(
                f"{', '.join(shared)} change code other models run: a regression check follows them"
            )
        # What asks the planner for nothing is answered by a revision that holds the tasks.
        self._note("; ".join(said), added=() if unplaced else tuple(t.id for t in added))

    def _admission(self) -> dict[str, Any]:
        """What a planner's graph is validated against; its shared code is the loop's to follow."""
        return {k: v for k, v in self.validation.items() if k != "own_paths"}

    def _follow_shared_changes(self) -> None:
        """Follow every change to shared code with a regression check (:func:`~.graph.add_regression_check`).

        The loop's, not the planner's: a task that changes code other models
        run is added to the pending regression check, or a new one is added
        when none is pending. The planner hears of a check it did not write.
        """

        def wiring() -> list[tuple[str, tuple[str, ...]]]:
            return [
                (t.id, tuple(t.depends_on))
                for t in self.graph.tasks
                if graph_mod.is_regression_check(t)
            ]

        before = wiring()
        added = graph_mod.add_regression_check(self.graph, self.validation["own_paths"])
        if added is None:
            if wiring() != before:
                self._save_graph()
            return
        if self._planner is not None:
            self._added_in_turn.add(added.id)
        self.graph.revisions.append(
            graph_mod.Revision(
                len(self.graph.revisions) + 1,
                f"orchestrator: {added.id} checks the changes to shared code "
                f"{', '.join(added.depends_on) or 'that landed'} make",
            )
        )
        self._note(
            f"the orchestrator added {added.id}, a regression check with no agent turn: once "
            f"{', '.join(added.depends_on) or 'it can'} land(s), it runs the modeling_v2 package "
            f"tests on the changed shared code; a failure comes back to you",
            added=(added.id,),
        )
        self._save_graph()

    def _promote(self, promotion=None) -> str:
        """Add a promote task when one is due; what keeps the target from delivery, or "".

        Entry by entry, once its local variants have landed and the target
        imports them, a promote merges them into the catalog
        (``operators/promotion.py``). A check task that has not started waits for
        it, so it measures the merged entries rather than the variants.
        """
        if self.s.promotion is None:
            return ""
        promotion = promotion if promotion is not None else self.s.promotion(self.graph)
        if not promotion.entries:
            return ""
        if promotion.task is None:
            return promotion.waiting
        task = graph_mod.Task.from_dict(promotion.task)
        self.graph.tasks.append(task)
        # A check that has not started would measure the variants the promote
        # is about to replace, and be stale once it lands: it waits for it.
        for check in self.graph.tasks:
            if (
                check.kind == graph_mod.CHECK_KIND
                and check.status == "pending"
                and task.id not in check.depends_on
            ):
                check.depends_on.append(task.id)
        self.graph.revisions.append(
            graph_mod.Revision(
                len(self.graph.revisions) + 1,
                f"orchestrator: {task.id} promotes the target's local variants of "
                f"{', '.join(promotion.entries)}",
            )
        )
        if self._planner is not None:
            self._added_in_turn.add(task.id)
        self._note(
            f"the orchestrator added {task.id} ({task.ref}), which starts now: it merges the "
            f"target's local variants into the shared catalog, and a regression check "
            f"follows it",
            added=(task.id,),
        )
        self._save_graph()
        return f"{task.id} promotes them"

    def _timeout(self) -> float | None:
        """How long to wait for something in flight to end before looking again (None: until then)."""
        # The operator's messages end no future: look for them while a long check runs.
        timeout = self.poll_s if self.s.stage_notes is not None else None
        if graph_mod.ready(self.graph, self._receipts_landed):
            timeout = self.poll_s  # ready work starts as devices free up
        if (
            self.events
            and self._planner is None
            and not self._urgent
            and self._first_event_at is not None
        ):
            left = self._first_event_at + self.event_window_s - time.monotonic()
            timeout = left if timeout is None else min(timeout, left)
        if self._planner is None and self._planner_retry_at > time.monotonic():
            left = self._planner_retry_at - time.monotonic()
            timeout = left if timeout is None else min(timeout, left)
        return None if timeout is None else max(timeout, 0.0)

    def run(self) -> str:
        """Drive the run to a verdict: delivered, escalated, ceiling, planner-stalled or planner-refused.

        Or ``stopped: <reason>``, when :func:`stoppable` asked it to stop: it
        then returns at once, and the turns and checks in flight are left to
        end on their own. The loop's own failure, or Ctrl-C, records what
        finished and raises at once, the same way; a resumed run starts them again.
        """
        checks = ThreadPoolExecutor(
            max_workers=self.max_parallel_checks, thread_name_prefix="mv2-check"
        )
        tasks = ThreadPoolExecutor(
            max_workers=max(1, self.max_parallel), thread_name_prefix="mv2-task"
        )
        planner = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mv2-planner")
        self._check_pool = checks
        crashed = False
        try:
            try:
                return self._drive(tasks, planner)
            except (Exception, KeyboardInterrupt):
                # The loop's own failure, or Ctrl-C: what finished is recorded before the process
                # goes, and nothing still running is waited for; a resumed run starts it again.
                crashed = True
                self._record_finished()
                raise
            finally:
                # A task thread waiting on its soft dependencies would wait for ever.
                with self._ended:
                    self._closing = True
                    self._ended.notify_all()
                stopped = crashed or self._stop.done()
                for pool in (planner, tasks):
                    pool.shutdown(wait=not stopped, cancel_futures=stopped)
        finally:
            # After the tasks: a task still running submits its checks here.
            stopped = crashed or self._stop.done()
            checks.shutdown(wait=not stopped, cancel_futures=stopped)
            self._check_pool = None

    def _record_finished(self) -> None:
        """Record the tasks whose thread already ended, as the loop would have: their work stands."""
        for tid, future in list(self._running.items()):
            task = self.graph.by_id().get(tid)
            if task is None or not future.done():
                continue
            try:
                decision, failed = future.result()
            except Exception as exc:  # recorded as the run's own failure: retried on resume
                decision, failed = f"raised {type(exc).__name__}: {exc}", []
            try:
                self._finish(task, decision, failed)
            except Exception as exc:  # what could not be recorded runs again on resume
                self.s.log(f"[loop] {tid}'s outcome could not be recorded: {exc}")

    def _drive(self, tasks: ThreadPoolExecutor, planner: ThreadPoolExecutor) -> str:
        running: dict[str, Future] = {}
        self._running = running
        # A resumed run whose contract held before it stopped keeps a milestone for it.
        self._correctness(*self._evaluated())
        completion = None
        if not self.graph.tasks and not self._stop.done():
            self._wake_planner(planner)
        while True:
            if self._stop.done():
                self.s.log(f"[stop] {self._stop.result()}: nothing more starts")
                return f"stopped: {self._stop.result()}"
            for tid, future in list(running.items()):
                if future.done():
                    del running[tid]
                    task = self.graph.by_id()[tid]
                    try:
                        decision, failed = future.result()
                    except Exception as exc:  # the run's own failure, not the task's: retried
                        decision, failed = f"raised {type(exc).__name__}: {exc}", []
                    self._finish(task, decision, failed)
            if self._settling is not None and all(f.done() for _, f in self._settling[1]):
                self._settled()
            for note in self.s.stage_notes() if self.s.stage_notes is not None else ():
                self._note(note, urgent=True)
            if self._planner is not None and self._planner.done():
                verdict = self._planner_returned(planner)
                if verdict:
                    return verdict
            if not self._written():
                # A full filesystem: what runs goes on and is recorded here; nothing starts.
                self._wait(running, self.poll_s)
                continue
            free = self.s.free_gpus()
            gated = self._gated()
            # A task takes a slot only while its thread needs an agent: a check
            # task never does, nor one waiting on its soft dependencies or running
            # its checks, nor a campaign, whose turns use only idle slots.
            kinds = {t.id: t.kind for t in self.graph.tasks}
            workers = sum(
                1
                for tid in running
                if kinds.get(tid) != graph_mod.CHECK_KIND
                and kinds.get(tid) not in BACKGROUND_KINDS
                and tid not in self._agentless
            )
            # One agent slot stays free for the planner (agents/slots.py).
            slots = max(AgentGate(self.max_parallel).task_slots - workers, 0)
            self._resumed &= {t.id for t in self.graph.tasks if t.status == "pending"}
            self._follow_shared_changes()
            waiting = self._regression_step() | self._closing_holds()
            ready_now = {t.id for t in graph_mod.ready(self.graph, self._receipts_landed)}
            now = time.monotonic()
            self._ready_since = {tid: self._ready_since.get(tid, now) for tid in ready_now}
            chosen = graph_mod.next_to_start(
                self.graph,
                free,
                gated,
                resumed=self._resumed | self._waited_for(),
                held=self._held,
                held_back=waiting,
                receipts_landed=self._receipts_landed,
                waited={tid: (now - since) / 60.0 for tid, since in self._ready_since.items()},
            )
            started = [t for t in chosen if t.kind == graph_mod.CHECK_KIND] + [
                t for t in chosen if t.kind != graph_mod.CHECK_KIND
            ][:slots]
            for task in started:
                self._record_batch(task)
                task.status = "running"
                self._resumed.discard(task.id)
                if task.kind == graph_mod.CHECK_KIND:
                    running[task.id] = self._start_check_task(task)
                else:
                    running[task.id] = tasks.submit(self._run_task, task)
            if started:
                self._save_graph()
            self._foreground_demand = (
                workers
                + len(
                    [t for t in started if t.kind not in BACKGROUND_KINDS | {graph_mod.CHECK_KIND}]
                )
                + len(
                    [
                        t
                        for t in graph_mod.ready(self.graph, self._receipts_landed)
                        if t.priority == "required"
                        and t.kind not in BACKGROUND_KINDS | {graph_mod.CHECK_KIND}
                    ]
                )
            )
            # The target is kept verified as it changes: beside foreground work the
            # riskiest stale checks and the delivery rehearsals, and once only a
            # campaign runs (it changes nothing a check measures) the rest.
            if self._settling is None:
                self._settle(busy=bool(self._foreground(running)))
            idle = not running and self._settling is None and self._planner is None
            # Nothing in flight but a campaign: what the graph owes is read as
            # when idle, for the campaign gives way once full support holds.
            spent = idle or (
                not self._foreground(running) and self._settling is None and self._planner is None
            )
            if spent:
                marker, evaluation = self._evaluated()
                promotion = self.s.promotion(self.graph) if self.s.promotion else None
                holds = evaluation.holds and not self._required_open(campaign=False)
                unpromoted = self._promote(promotion)
                if holds and unpromoted:
                    if unpromoted != self._unpromoted_said:
                        self._unpromoted_said = unpromoted
                        self._note(
                            f"the contract holds, but local variants wait for promotion: "
                            f"{unpromoted}",
                            urgent=True,
                        )
                    holds = False
                completion = self._completion(
                    marker, evaluation, unpromoted=bool(promotion and promotion.entries)
                )
                self._correctness(marker, evaluation, completion)
                unreviewed, gaps = completion.unreviewed, completion.gaps
                if completion.ready:
                    if idle:
                        return "delivered"
                    self._stop_campaigns(running)
                # Standing complaints wake no turn while an escalation waits on the operator.
                if unreviewed and not self._escalated:
                    self.revisions_since_progress += 1
                    self._note(
                        "every required check holds, but no regression check covers "
                        f"{', '.join(unreviewed)}, which landed changes to code other models "
                        "run: the run is not delivered before one has passed",
                        urgent=True,
                    )
                elif idle and self._done_said and not holds:
                    self.revisions_since_progress += 1
                    self._note(
                        "planner said DONE, but the contract does not hold:\n"
                        + evaluation.render(),
                        urgent=True,
                    )
                elif idle and self._done_said and gaps:
                    # Every check holds, and something else keeps full support out of
                    # reach: a DONE that changes nothing is no progress either.
                    self.revisions_since_progress += 1
                    self._note(
                        "planner said DONE, but full support does not hold:\n"
                        + "\n".join(f"- {gap}" for gap in gaps),
                        urgent=True,
                    )
                if idle:
                    self._done_said = False
                    blocked = graph_mod.ready(self.graph, self._receipts_landed)
                    if blocked and not self._escalated:
                        # Ready work that cannot start with nothing running never will.
                        reasons = ", ".join(
                            f"{t.id} (waits for {', '.join(gated[t.kind])})"
                            if t.kind in gated
                            else f"{t.id} ({self._held(t)} GPUs)"
                            for t in blocked
                        )
                        self._note(
                            f"ready tasks cannot start on {free} free GPUs: {reasons}", urgent=True
                        )
                        self.revisions_since_progress += 1
                self._save_events()
            if self._planner is None:
                if self._escalated and idle and not self._loud():
                    # Nothing runs, and nothing came in since: only the operator can move it.
                    return "escalated"
                if self._loud() and self._first_event_at is None:
                    self._first_event_at = time.monotonic()
                # With nothing in flight the graph is spent or stuck: the planner
                # decides what comes next. A campaign alone in flight leaves the
                # graph as idle for hours, so its events do not wait for the window.
                # Otherwise the planner reads events as they come due.
                background = bool(self.events) and self._only_background(running)
                retry_due = time.monotonic() >= self._planner_retry_at
                if retry_due and (idle or background or self._events_due()):
                    # A stall is judged only once no task but a campaign runs: while
                    # work is in flight its outcome may still be the progress.
                    if spent and graph_mod.stalled(
                        self.revisions_since_progress, self.revision_cap
                    ):
                        # The contract holds and only the delivery checks on disk fail:
                        # the delivery phase names each, as its verdict.
                        return (
                            "delivery-blocked"
                            if completion is not None and completion.files_only
                            else "planner-stalled"
                        )
                    self._wake_planner(planner)
            self._wait(running, self._timeout())

    def _wait(self, running: dict[str, Future], timeout: float | None) -> None:
        """Until something in flight ends, the run is stopped, or ``timeout`` passes.

        With nothing in flight, the timeout (the poll interval when there is none).
        """
        in_flight = self._in_flight(running)
        if not in_flight and timeout is None:
            timeout = self.poll_s
        wait([*in_flight, self._stop], timeout=timeout, return_when=FIRST_COMPLETED)

    def _in_flight(self, running: dict[str, Future]) -> list[Future]:
        """The tasks, re-checks and planner turn in flight, as futures to wait on.

        A task or planner turn that ended since the loop looked
        ends the wait at once, and is recorded. A batch of re-checks is read
        once all of it is in, so until then only its members still out are
        waited on: one already in would end every wait at once.
        """
        in_flight = list(running.values())
        if self._settling is not None:
            batch = [f for _, f in self._settling[1]]
            in_flight += [f for f in batch if not f.done()] or batch
        if self._planner is not None:
            in_flight.append(self._planner)
        return in_flight


def seed(workspace: Path | str, tasks: Sequence[graph_mod.Task], reason: str, event: str) -> bool:
    """Start a run that has no graph yet with ``tasks``, beside the planner's first turn.

    Writes ``graph.yaml`` holding them, as the orchestrator's own revision
    (``reason``), and leaves ``event`` for the planner: the loop wakes it at
    once and starts the tasks beside its turn, with no turn before them. A
    workspace that has a graph is left as it is. Returns whether it was seeded.
    """
    workspace = Path(workspace)
    path = workspace / graph_mod.GRAPH_FILENAME
    if path.exists():
        return False
    events = workspace / EVENTS_FILENAME
    saved = json.loads(events.read_text(encoding="utf-8") or "{}") if events.is_file() else {}
    # The event first: a graph on disk is what says the run was seeded.
    if event not in (saved.get("events") or ()):
        saved["events"] = [*(saved.get("events") or ()), event]
        write_text_atomic(events, json.dumps(saved, indent=1))
    graph = graph_mod.Graph(
        tasks=list(tasks), revisions=[graph_mod.Revision(1, f"orchestrator: {reason}")]
    )
    graph.save(path)
    return True


def _kernel_task_id(item: vocab_mod.QueueItem, taken: Iterable[str]) -> str:
    """``k<row>-<entry name>`` for a queue row's kernel task, numbered on when that is taken."""
    taken = set(taken)
    stem = f"k{item.n}-{item.target.rstrip('/').rsplit('/', 1)[-1].replace('_', '-')}"
    name, n = stem, 1
    while name in taken:
        n += 1
        name = f"{stem}-{n}"
    return name


def _check_task_event(task: graph_mod.Task, failed: Sequence[str]) -> str:
    """What the planner hears of a ``check`` task: its checks ran, with no agent turn."""
    if task.regression:
        said = (
            f"task {task.id} (regression check) ran the package tests after the changes to "
            f"shared code it covers ({', '.join(task.covers or ()) or 'none'})"
        )
        if not failed:
            return f"{said}; they passed"
        return (
            f"{said}; what failed: {'; '.join(failed)}. No agent has looked: an `investigate` "
            "task finds why; a fix that changes shared code is followed by the next regression "
            "check, or retry this one (`status: pending`) once what it ran on has changed"
        )
    said = f"task {task.id} (check) ran its checks with no agent turn"
    if not failed:
        return f"{said}; they passed"
    return (
        f"{said}; checks failed: {', '.join(failed)}. No agent has looked: an `investigate` "
        "task finds why, or retry it (`status: pending`) once what it measures has changed"
    )


def _answered(
    outcome: _Outcome,
    written: Mapping[str, tuple[str, int]],
    adopted: Mapping[str, graph_mod.Task],
) -> bool:
    """Whether a revision re-planned the task past the run ``outcome`` reports.

    It dropped the task (``written`` does not name it); it retried it, so the
    task's attempt in ``adopted`` is past the event's; it reopened an
    abandoned one; or, for a run that ended done, it wrote the task done at
    that attempt, from a graph that already held the result. A failure the
    revision kept as failed is not answered: a turn that copies the graph
    copies a failure it never read.
    """
    if outcome.task not in written:
        return True
    task = adopted.get(outcome.task)
    if task is not None and task.attempts > outcome.attempt:
        return True
    if outcome.status == "abandoned":
        return task is not None and task.status == "pending"
    return outcome.status == "done" and written[outcome.task] == ("done", outcome.attempt)


@contextlib.contextmanager
def stoppable(workspace: Path | str) -> Iterator[Callable[[str], None]]:
    """While open, the callable it yields stops the run this process drives on ``workspace``.

    Called with the reason, from any thread, it makes the loop start nothing
    more and return ``stopped: <reason>`` at once; the turns and checks in
    flight are left to end on their own and their outcomes are never read. A
    loop that starts after it was called stops as it starts. For what runs
    beside the run's first turns and can find the run cannot go on, such as
    preflight's substrate gate.
    """
    key = Path(workspace).resolve()
    stopped: Future = Future()

    def stop(reason: str) -> None:
        with _STOPS_LOCK:
            if not stopped.done():
                stopped.set_result(reason)

    with _STOPS_LOCK:
        _STOPS[key] = stopped
    try:
        yield stop
    finally:
        with _STOPS_LOCK:
            if _STOPS.get(key) is stopped:
                del _STOPS[key]


__all__ = [
    "CEILING",
    "CHECKED",
    "DEFAULT_MAX_PARALLEL_CHECKS",
    "DEFAULT_MAX_PARALLEL_TASKS",
    "DONE",
    "ESCALATE",
    "FINISHED",
    "GraphOrchestrator",
    "PlannerAnswer",
    "PlannerTurn",
    "Services",
    "seed",
    "stoppable",
]
