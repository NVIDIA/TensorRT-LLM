# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The graph-driven run loop (docs/design.md sections 4 to 6).

The planner writes the task graph; this loop admits it, starts tasks as
devices and paths allow, runs each finished task's checks itself, and asks
the planner again when something happens. The run ends when the contract
holds, when the planner escalates or hits a ceiling, or when it stalls.

Nothing in the loop waits for what it does not need. Each task's worker runs
in a thread of its own; every check, a task's or a re-check of an idle run,
runs on one bounded pool, as many at once as it allows; and the planner's
turn runs in a thread beside them, one turn at a time, while the loop keeps
finishing and starting tasks. The critic reads a reshaped graph on a thread
of its own, and nothing waits for its words. Only the loop's own thread
changes the graph or writes it: a planner's revision is applied when its turn
returns, against the graph as it is by then.

Work whose shape no planner turn decides starts without one: a new run starts
with the survey (:func:`seed`), and when the survey lands the loop adds one
kernel task per vocabulary queue row itself. Events a revision already
answered do not wake the planner again.

Work that needs no agent runs with none. A ``check`` task's checks go to the
check pool the moment its dependencies are done; it takes no task slot and
no devices of its own, and only its failure reaches the planner. A task that
soft-depends on another starts once that one has started and waits for it
only at its integration, until that one's commits are integrated
(:meth:`GraphOrchestrator.soft_dependencies`); meanwhile, and while a task's
checks run, it holds no task slot and no devices. Agent slots go by priority
(``infra/execution/agent_gate.py``): the planner first, then the critical
path, the other tasks, the critic and the improver, and a campaign's turns
only on slots the foreground work leaves idle.

Cross-model regression reviews are batched. A review whose dependencies are
done waits while a task that changes shared code runs, for at most the
regression window after the first change of its batch landed, and while
another regression review runs; it then covers every change to shared code
landed that no review covered (:func:`~.graph.regression_batch`). One whose
batch is empty when it would start has nothing left to review and closes
with no turn. The run is not delivered while a landed change to shared code
is covered by no review.

The target is kept verified as it changes (:meth:`GraphOrchestrator._settle`):
each landing runs the riskiest checks it made stale again beside the work in
flight, a member's delivery tier is rehearsed as soon as its other checks
hold, and once no required task is open the remaining stale checks run, no
optional change lands meanwhile, and the delivery tier runs on the target as
it is.

Correctness does not wait for performance. A tune campaign runs for hours
beside the graph on capacity nothing else needs and changes nothing a check
measures. Full
support is one predicate (:meth:`GraphOrchestrator._support_gaps`): the
contract holds on the target as it is, the delivery files on disk check out,
no landed change to shared code waits for a regression review, and no local
variant waits for promotion. The first state on which it holds is recorded as
the correctness milestone, with no planner turn
(:meth:`GraphOrchestrator._correctness`); at that point a running campaign is
told to wrap up, its best branch is reported as an artifact of the run and
never integrated, and the run is delivered.

A reload request (``infra/driver/reload.py``) drains the loop: no task,
integration, check or planner turn starts; what is in flight runs to its end
(``run/drain.py`` keeps the door and the ledger); then the loop checkpoints and
the workflow re-executes the process on the code now at its checkout. Worker
turns are not waited for, as a relaunched process adopts them.

The workflow's services arrive as callables (:class:`Services`), so the loop
runs under fakes on CPU exactly as it runs under the workflow.
"""

from __future__ import annotations

import contextlib
import copy
import json
import os
import socket
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, ContextManager, Iterable, Iterator, Mapping, Sequence

from ..agents import decisions as decision_table
from ..infra.driver import reload as reload_mod
from ..infra.execution.agent_gate import AgentGate, Priority, run_at, turn_priority
from ..onboard import vocabulary as vocab_mod
from . import contract as contract_mod
from . import drain as drain_mod
from . import graph as graph_mod
from .integration import INTEGRATED_KINDS

#: Planner decisions.
GRAPH_UPDATED, DONE, ESCALATE, CEILING = "GRAPH_UPDATED", "DONE", "ESCALATE", "CEILING"
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
DEFAULT_MAX_PARALLEL_TASKS = 8
DEFAULT_MAX_PARALLEL_CHECKS = 8
#: Task kinds that run for hours beside the graph: while only they run, the
#: planner reads its events at once instead of waiting for the event window.
BACKGROUND_KINDS = graph_mod.BACKGROUND_KINDS
#: How many times a task the run's infrastructure failed starts again on its own.
INFRA_RETRIES = 1
#: How many stale checks run again, riskiest first, beside running work each time
#: the target changes; the rest wait for the graph to run out of work.
RECHECKS_WHILE_BUSY = 4
#: How long a task thread waiting on its soft dependencies sleeps between looks,
#: past the wake-up each recorded outcome gives it.
SOFT_POLL_S = 5.0
#: Once a drain holds nothing, how long it waits for a task or re-check whose last
#: check just ended to hand its outcome in, so that it is recorded before the re-exec.
DRAIN_SETTLE_S = 0.5

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
    #: Where the cross-model regression reviews stand: the batch the next one
    #: takes, and the reviews running and waiting. Empty when the run does not bind the rule.
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
class _Mark:
    """What the loop knows of an event besides its words."""

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
    #: What a task holds of those devices while it runs (``capacity.held_gpus``).
    held_gpus: Callable[[graph_mod.Task], int] = lambda task: task.gpus
    checkpoint: Callable[[], None] = lambda: None
    log: Callable[[str], None] = lambda message: None
    #: The target's identity now; a verdict recorded under another is stale.
    marker: Callable[[], str] = lambda: ""
    #: An advisory read of a newly adopted graph, on a thread of its own: nothing waits for
    #: it, and its words reach the planner's next turn.
    critique: Callable[[graph_mod.Graph, str], str] | None = None
    #: The loop has recorded how a task's turn ended (done, failed, abandoned, or pending
    #: again after a failure of the infrastructure): its answer is consumed.
    finished: Callable[[graph_mod.Task], None] = lambda task: None
    #: Whether a task whose turn did not finish landed commits on the target all the
    #: same, as a failed receipt's guard tests do; one that finished always has.
    landed: Callable[[graph_mod.Task], bool] = lambda task: False
    #: Holds a check until the sessions' free devices let it start, and while
    #: it runs (one context manager per check); without it a check starts as
    #: soon as the check pool has a thread for it.
    admit: Callable[[contract_mod.Check], ContextManager[None]] | None = None
    #: Replace this process with the code now at the checkout, once a reload drained; given
    #: the reload's record. It does not return when it did: a return or an ``OSError``
    #: leaves the run going on here. ``None``: this process cannot, and requests wait.
    reload: Callable[[dict], None] | None = None
    #: Every required check holds on the target as it is (its marker, and the evaluation):
    #: name the commit it is at and journal it; returns ``{"sha", "ref", "elapsed_s"}``, or
    #: ``None`` when no commit holds what the checks measured. ``None``: no milestone is kept.
    correct: Callable[[str, contract_mod.Evaluation], Mapping[str, Any] | None] | None = None
    #: The target's local variants of shared entries, given the graph: a
    #: ``promotion.Promotion`` naming the entries waiting, and the promote task to add
    #: now or why none is (``onboard/promotion.py``). ``None``: nothing is promoted.
    promotion: Callable[[graph_mod.Graph], Any] | None = None
    #: The contract evaluated on the target (its marker, and the evaluation), before
    #: ``correct``: name the bring-up's commit the first time its checks hold. ``None``:
    #: no bring-up is named.
    bringup: Callable[[str, contract_mod.Evaluation], Any] | None = None
    #: The delivery checks on disk (the CI accuracy test, the test list entry, the
    #: accuracy reference, the target's files and commits), as cheap as reading files:
    #: what fails, one line each; empty when all pass. ``None``: none are checked here.
    delivery_files: Callable[[], list[str]] | None = None
    #: Tell a running campaign to wrap up: full support holds, so it dispatches nothing
    #: new and ends with its best kept tree. Given why.
    stop_campaign: Callable[[str], None] = lambda reason: None
    #: The task that writes the delivery files (the CI accuracy test, its test list
    #: entry, the accuracy reference), given the graph: as ``Task.from_dict`` reads one,
    #: or ``None`` when the graph already has one or it cannot be written yet.
    delivery_task: Callable[[graph_mod.Graph], Mapping[str, Any] | None] | None = None
    #: The tasks that fix engine defects workers reported since last asked, given the
    #: graph: as ``Task.from_dict`` reads them, none the graph already has.
    defect_tasks: Callable[[graph_mod.Graph], list[Mapping[str, Any]]] | None = None
    #: What the stage learned for the planner outside any task since last asked (a
    #: probe of the target's geometry, say): each an event of the next turn.
    stage_notes: Callable[[], list[str]] | None = None
    #: What a gate result read of the tree, as the contract records it
    #: (:attr:`~.contract.Verdict.reads`, ``reads_marker``).
    narrow: Callable[[Any], tuple[str, str]] | None = None
    #: Whether a verdict recorded on another state of the target holds on it as it
    #: is now: what it read of the tree is unchanged.
    holds: Callable[[contract_mod.Verdict], bool] | None = None


#: How a full-support gap of the delivery checks on disk begins (``_support_gaps``).
DELIVERY_FILES_GAP = "delivery files: "
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


def _write_whole(path: Path, text: str) -> None:
    """Replace ``path`` in one step, so a reader never sees half of it."""
    partial = path.with_name(path.name + ".partial")
    partial.write_text(text, encoding="utf-8")
    os.replace(partial, path)


class ChecksRaised(Exception):
    """Some checks raised instead of returning a verdict (a session was lost, say).

    Their outcome is unknown, not a failure: ``failed`` are the keys that did
    fail, ``raised`` what each of the others raised.
    """

    def __init__(self, failed: list[str], raised: Mapping[str, str]) -> None:
        super().__init__("; ".join(f"{key}: {why}" for key, why in raised.items()))
        self.failed = failed
        self.raised = dict(raised)


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
        regression_window_s: float = graph_mod.REGRESSION_WINDOW_S,
        repo_root: str | Path = "",
    ) -> None:
        self.s = services
        self.contract = contract
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
        #: The door a reload closes, and the ledger of what is in flight (``inflight.json``).
        self.inflight = drain_mod.InFlight(workspace)
        # A reload asked of a process that is gone is answered by this one's start.
        answered = reload_mod.read_request(workspace)
        if answered and float(answered.get("requested_at") or 0) >= drain_mod.PROCESS_STARTED_AT:
            answered = None
        if answered:
            reload_mod.remove_request(workspace)
        #: What the stop before this start cost (:func:`~.drain.restart_record`).
        self.restart = drain_mod.restart_record(
            self.inflight.previous,
            reset=[t.id for t in restarted],
            answered_request=answered,
        )
        record = reload_mod.read_record(workspace)
        if record and record.get("phase") == reload_mod.PHASE_DRAINING:
            # Its process stopped before it could re-execute.
            self._write_reload_record({**record, "phase": reload_mod.PHASE_ABANDONED})
        #: Tasks that were running when the last process stopped and have not
        #: started again: they start ahead of every other ready task.
        self._resumed = {t.id for t in restarted}
        if restarted:
            # On disk too: the planner reads graph.yaml, and a task it sees
            # running is one it believes a worker has.
            self._save_graph()
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
        #: How long a ready regression review waits for more shared changes to land.
        self.regression_window_s = regression_window_s
        #: Events the planner has not read yet. On disk, so a process that dies
        #: while they wait (or during the planner turn that reads them) hands
        #: them to the next one instead of dropping a critique or a report.
        self.events_path = Path(workspace) / EVENTS_FILENAME
        saved: dict[str, Any] = {}
        if self.events_path.is_file():
            saved = json.loads(self.events_path.read_text(encoding="utf-8") or "{}")
        self.events: list[str] = [str(e) for e in saved.get("events") or ()]
        #: Per event, in step with :attr:`events`: whether it is urgent and which
        #: task run it reports the end of. What a dead process left is urgent and
        #: reaches the next turn whole, so none of it is kept on disk.
        self._marks: list[_Mark] = [_Mark()] * len(self.events)
        #: A worker reported something the planner has not read yet. The run is
        #: not delivered past it: a tune task's best branch may still need merging.
        self.unread_reports = bool(saved.get("unread_reports"))
        #: Per task, how many times it started again after the infrastructure failed it.
        self.retries: dict[str, int] = {
            str(k): int(v) for k, v in (saved.get("retries") or {}).items()
        }
        #: Per task whose commits landed on the target, when its latest landing was (in
        #: seconds since the epoch) and which attempt it was: the regression window runs
        #: from the first landing of a batch, and a failed task's landing joins one.
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
        #: The critic's read in flight, and the latest reshaped graph waiting for
        #: it, with the reason it was written.
        self._critic: Future | None = None
        self._critic_next: tuple[graph_mod.Graph, str] | None = None
        #: The turn in flight: how many of the events it was given, which attempt
        #: it is, and whether a worker reported while it ran.
        self._turn_events = 0
        self._turn_attempt = 0
        self._reports_in_turn = False
        #: The planner said DONE; it is told otherwise once nothing is left to run or check.
        self._done_said = False
        #: Why local variants last kept a holding contract from delivery, as the planner heard it.
        self._unpromoted_said = ""
        #: The delivery files that last kept a holding contract from full support, per
        #: marker, as the planner heard them.
        self._files_said: tuple[str, tuple[str, ...]] | None = None
        #: The delivery files read last, and the marker of the target they were read on.
        self._files_read: tuple[str, list[str]] | None = None
        #: The contract held when last read, and only the delivery files kept full support out.
        self._files_alone = False
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
        self._passed_seen = len(self._evaluation().passed)
        #: Where correctness was last delivered (``correctness.jsonl``), and the failures on
        #: a later state of the target last journaled against it.
        self.milestone = contract_mod.latest_milestone(workspace)
        self._regressed: tuple[str, tuple[str, ...]] | None = None
        #: The reload under way: when it started, its deadline, what it waits for.
        self._drain: dict[str, Any] | None = None
        self._watch: drain_mod.RequestWatch | None = None
        #: A request whose re-exec failed, as it was then: asked again, it is tried again.
        self._failed_request: tuple | None = None
        #: Done once the run is asked to stop (:func:`stoppable`); its result says why.
        with _STOPS_LOCK:
            self._stop: Future = _STOPS.get(self.workspace.resolve()) or Future()
        #: Notified whenever a task's outcome is recorded or the graph is revised, for
        #: the task threads waiting on soft dependencies; ``_closing`` once the loop ends.
        self._ended = threading.Condition()
        self._closing = False

    def _save_events(self) -> None:
        state = {
            "events": self.events,
            "unread_reports": self.unread_reports,
            "retries": self.retries,
            "landings": {k: list(v) for k, v in self._landings.items()},
            "receipts_landed": self._receipts_landed,
        }
        _write_whole(self.events_path, json.dumps(state, indent=1))

    def _save_graph(self) -> None:
        """Write ``graph.yaml`` whole: a planner turn may be reading it."""
        partial = self.graph_path.with_name(self.graph_path.name + ".partial")
        self.graph.save(partial)
        os.replace(partial, self.graph_path)

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
        marks = self._aligned_marks()
        self.events.append(event)
        self._marks = [*marks, _Mark(urgent, outcome, report, quiet, tuple(added))]
        if report:
            self.unread_reports = self._reports_in_turn = True
        if self._first_event_at is None and not quiet:
            self._first_event_at = time.monotonic()
        self._urgent = self._urgent or urgent

    def _aligned_marks(self) -> list[_Mark]:
        """:attr:`_marks` in step with :attr:`events`: an event written there directly is urgent."""
        marks = self._marks[: len(self.events)]
        return marks + [_Mark()] * (len(self.events) - len(marks))

    # ----------------------------------------------------------------- checks
    def _check_by_key(self, key: str) -> contract_mod.Check | None:
        return next((c for c in self.contract.checks if c.key == key), None)

    def _delivery_keys(self) -> set[str]:
        return {c.key for c in self.contract.by_tier(contract_mod.TIER_DELIVERY)}

    def _run_check(self, check: contract_mod.Check) -> bool:
        """One check, in a pool thread, recorded under the target as it was when it started.

        A check the contract already counts as passed on this very target (its
        own verdict, or an equivalent check's) does not run again. Whether it
        does is asked as the check starts, so of two equivalent checks the
        later one to start reads the earlier one's verdict once it is in.

        A check is in flight from its start to its verdict (:attr:`inflight`):
        one that would start while a reload drains waits instead, and the
        process is replaced before it does. A check waiting for devices
        (:attr:`Services.admit`) has started nothing, so a drain does not wait
        for it; the target is read once it is admitted.
        """
        admitted = self.s.admit(check) if self.s.admit is not None else contextlib.nullcontext()
        with admitted, self.inflight.hold("check", check.key):
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
                self.status.record(
                    check.unit, result, marker=marker, inputs=check.inputs, reads=reads
                )
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
        """The priority ``task``'s turns take agent slots at (``agent_gate.Priority``).

        A campaign's are background; a required task on the longest open
        critical path is critical; other required tasks are required, and a
        ``nice_to_have`` one waits beside the critic and the improver.
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
                # What changed last, for the order stale checks run again in (:meth:`_by_risk`).
                self._landed, self._landed_marker = task, self.s.marker()
            self._ended.notify_all()

    def _evaluated(self) -> tuple[str, contract_mod.Evaluation]:
        """The target's marker now, and the contract evaluated on it."""
        marker = self.s.marker()
        with self._status_lock:
            return marker, contract_mod.evaluate(
                self.contract, self.status, marker=marker, holds=self.s.holds
            )

    def _evaluation(self) -> contract_mod.Evaluation:
        return self._evaluated()[1]

    def _support_gaps(self, marker: str, evaluation: contract_mod.Evaluation) -> list[str]:
        """What keeps the target as it is from full support; empty when support is complete.

        One predicate for the correctness milestone, the run's delivery and the
        campaign's end: every required check holds on the target as it is, the
        delivery files on disk check out (:attr:`Services.delivery_files`), no
        landed change to shared code waits for a cross-model regression review,
        and no local variant waits for promotion. The cheap ones are read only
        once the contract holds. A failing delivery file is told to the planner
        once per state of the target and set of failures.
        """
        if not evaluation.holds:
            return ["the contract does not hold"]
        gaps: list[str] = []
        unreviewed = self._unreviewed_landings()
        if unreviewed:
            gaps.append(f"no regression review covers {', '.join(unreviewed)}")
        if self.s.promotion is not None and self.s.promotion(self.graph).entries:
            gaps.append("local variants wait for promotion")
        missing = self._delivery_files(marker)
        if missing:
            gaps.append(DELIVERY_FILES_GAP + "; ".join(missing))
            said = (marker, tuple(missing))
            if said != self._files_said:
                self._files_said = said
                self._note(
                    "every required check holds, but the delivery files on disk do not check "
                    "out:\n" + "\n".join(f"- {m}" for m in missing) + "\nadd or fix the task "
                    "that writes them; the run is not delivered, and correctness is not "
                    "recorded, before they do",
                    urgent=True,
                )
                self._save_events()
        return gaps

    def _delivery_files(self, marker: str) -> list[str]:
        """:attr:`Services.delivery_files` on the target at ``marker``, read once per marker."""
        if self.s.delivery_files is None:
            return []
        if self._files_read is None or self._files_read[0] != marker:
            self._files_read = (marker, list(self.s.delivery_files()))
        return self._files_read[1]

    def _correctness(self, marker: str, evaluation: contract_mod.Evaluation) -> None:
        """Record correctness delivered on a state of the target that is fully supported.

        Once per state: the first one on which full support holds
        (:meth:`_support_gaps`), the delivery tier's checks included, becomes the
        milestone. The workflow names its commit (:attr:`Services.correct`) and
        it is appended to ``correctness.jsonl``, with no planner turn. Work
        that lands after it leaves it where it is; the next state full support
        holds on moves it forward. Required checks that fail on a later state
        are journaled against it, once per state and set of failures. An
        unestablished marker names no tree, and a contract none of whose checks
        passed holds nothing: no milestone is kept on either.
        """
        if not marker or not evaluation.passed:
            return
        if self.s.bringup is not None:
            self.s.bringup(marker, evaluation)
        if self.s.correct is None:
            return
        held = self.milestone
        if evaluation.holds:
            if held is not None and held.marker == marker:
                return
            if self._support_gaps(marker, evaluation):
                return
            named = self.s.correct(marker, evaluation)
            if not named:
                return
            elapsed = named.get("elapsed_s")
            self.milestone = contract_mod.Milestone(
                sha=str(named["sha"]),
                marker=marker,
                ref=str(named.get("ref") or ""),
                at=time.time(),
                elapsed_s=float(elapsed) if elapsed is not None else None,
                checks=len(evaluation.passed),
            )
            contract_mod.record_milestone(self.workspace, self.milestone)
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

        ``busy``, with foreground work running: each time a change lands,
        the :data:`RECHECKS_WHILE_BUSY` stale checks likeliest to fail
        (:meth:`_by_risk`) run again beside it, so a change that breaks
        another unit shows at once rather than at the end; and the delivery
        tier of a member whose other checks all hold is rehearsed once
        (:meth:`_rehearsal`), so a capacity limit shows before the last tree.
        Otherwise every stale check runs again; and once every other required
        check holds and no required task is open but a campaign, the delivery
        tier runs on the target as it is. Either runs on the check pool, so a
        task the planner adds meanwhile starts at once; :meth:`_settled` reads
        the verdicts once they are all in, and tells the planner what failed.
        """
        claimed = {k for t in self.graph.tasks if t.status == "done" for k in t.satisfies}
        delivery = self._delivery_keys()
        marker, evaluation = self._evaluated()
        stale = [c.key for c in evaluation.missing if c.key in claimed and c.key not in delivery]
        if busy or (not stale and self._required_open(campaign=False)):
            keys: list[str] = []
            # Once per landing the loop knows made the target what it is: the risk
            # order reads what that landing changed.
            landed = (marker, self._landed.id if self._landed is not None else "")
            known = self._landed is not None and self._landed_marker == marker
            if stale and known and landed != self._rechecked_at:
                self._rechecked_at = landed
                keys = self._by_risk(stale)[:RECHECKS_WHILE_BUSY]
            keys += self._rehearsal(evaluation)
            what = "checks run beside the work in flight failed on the current target: "
        else:
            what, keys = (
                "checks that held before failed on the current target: ",
                self._by_risk(stale),
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

    def _by_risk(self, keys: Sequence[str]) -> list[str]:
        """Stale checks in the order to run them again, those likeliest to fail first.

        First, when the change that landed last may move device memory, the
        checks of the configuration whose engine left the least free in
        ``runtime_facts.yaml`` (:meth:`_tightest`). Then the checks whose
        engines last left little memory free: those whose last verdict that
        read it warned ``LOW MARGIN``, and the one that left the least. The
        next change that adds memory to a configuration runs the tightest out
        first. Then the checks of the units the change that landed last
        reaches (:meth:`_units_reached`), then the rest; within each, the
        least memory free first, then the contract's order. The margins are
        read from the contract status
        (:meth:`~.contract.ContractStatus.margins`).
        """
        if not keys:
            return []
        first = self._tightest()
        with self._status_lock:
            margins = self.status.margins()
        free = {
            key: margins[key].min_free_gib
            for key in keys
            if key in margins and margins[key].min_free_gib is not None
        }
        tightest = min(free.values(), default=None)
        reached = self._units_reached()
        order = {c.key: i for i, c in enumerate(self.contract.checks)}

        def risk(key: str) -> tuple[int, float, int]:
            margin = margins.get(key)
            check = self._check_by_key(key)
            if key in first:
                return 0, float(first.index(key)), order.get(key, len(order))
            if (margin is not None and margin.low_margin) or (
                tightest is not None and free.get(key) == tightest
            ):
                rank = 1
            elif check is not None and check.unit in reached:
                rank = 2
            else:
                rank = 3
            return rank, free.get(key, float("inf")), order.get(key, len(order))

        return sorted(keys, key=risk)

    def _tightest(self) -> list[str]:
        """The checks to run again first: those of the configuration that left the least memory free.

        Only after a change that may move device memory landed (a path it
        touches is :func:`~.workflow.memory_relevant`); in the order of
        :func:`~.contract.tightest_checks`.
        """
        from . import runtime_facts
        from .workflow import memory_relevant

        landed = self._landed
        if landed is None or not any(memory_relevant(p) for p in landed.touches):
            return []
        return list(contract_mod.tightest_checks(self.contract, runtime_facts.load(self.workspace)))

    def _units_reached(self) -> set[str]:
        """The units whose checks the change that landed last may have moved.

        Its own: the unit it worked on (``ref``) and those of the checks it
        claims. And every unit one of whose tasks declares a path that is, or
        holds, or lies in a path the landed task declares. Nothing has landed
        yet in a process that has just started.
        """
        landed = self._landed
        if landed is None:
            return set()
        units = {c.unit for c in self.contract.checks}

        def own(task: graph_mod.Task) -> set[str]:
            claimed = {self._check_by_key(k) for k in task.satisfies}
            return {c.unit for c in claimed if c is not None} | ({task.ref} & units)

        reached = own(landed)
        for task in self.graph.tasks:
            if task.id != landed.id and graph_mod.paths_overlap(task.touches, landed.touches):
                reached |= own(task)
        return reached

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
            list(self.events),
            list(refusals),
            marker=marker,
            regression=self._regression_standing(),
        )
        self._planner = pool.submit(run_at, Priority.PLANNER, self.s.plan, turn)

    def _planner_returned(self, pool: ThreadPoolExecutor) -> str | None:
        """Apply what the planner's thread returned; a final answer is the run's verdict.

        A refused graph goes back at once, up to :data:`MAX_REFUSALS` turns.
        The first graph, and a revision that reshapes the graph, is adopted
        at once and read by the critic beside the run (:meth:`_critic_step`):
        nothing waits for its words, which reach the planner's next turn. The
        tasks the orchestrator added while the turn ran, which its graph
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
        if answer.decision in (ESCALATE, CEILING, DONE):
            self._consume_turn_events()
        if answer.decision == ESCALATE:
            self.s.log(f"[planner] escalated: {answer.summary}")
            return "escalated"
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
                self.graph, proposed, answer.summary, **self.validation
            )
            if not refusals:
                self._consume_turn_events()
                if self.s.critique is not None and _reshaped(self.graph, adopted):
                    # The latest reshaped graph is the one worth reading.
                    self._critic_next = (copy.deepcopy(adopted), answer.summary)
                self.graph = adopted
                self._added_in_turn = set()
                self._drop_answered(written)
                self._add_delivery_task()
                self._save_graph()
                self._wake_soft_waiters()
                self.revisions_since_progress += 1
                self.s.checkpoint()
                return None
            self.s.log("[planner] graph refused: " + "; ".join(refusals))
        if self._turn_attempt >= MAX_REFUSALS:
            return "planner-refused"
        # The events stay unread: the next turn reads them again, with why it was refused.
        self._wake_planner(pool, refusals)
        return None

    def _consume_turn_events(self) -> None:
        """The turn's answer was taken: the events it was given are read, those since are not."""
        marks = self._aligned_marks()
        del self.events[: self._turn_events]
        self._marks = marks[self._turn_events :]
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

    def _critic_step(self, pool: ThreadPoolExecutor) -> None:
        """Hand the critic's words to the planner once its read returns, and start the next read.

        The critic reads the latest graph a reshaping revision left waiting
        (:attr:`_critic_next`), one read at a time, beside the tasks and the
        planner's turns. A read that raised is logged: its words were advice.
        """
        if self._critic is not None and self._critic.done():
            future, self._critic = self._critic, None
            try:
                words = future.result().strip()
            except Exception as exc:  # an advisory read that raised costs the run nothing
                self.s.log(f"[critic] the read raised {type(exc).__name__}: {exc}")
                words = ""
            if words:
                self._note(f"critic (advisory) on this graph: {words}", urgent=True)
                self._save_events()
        if self._critic is None and self._critic_next is not None:
            graph, reason = self._critic_next
            self._critic_next = None
            self._critic = pool.submit(run_at, Priority.SIDE, self.s.critique, graph, reason)

    def _drop_answered(self, written: Mapping[str, tuple[str, int]]) -> None:
        """Drop the events of its turn that the revision just adopted answered.

        They arrived while the planner wrote it, and woke it again only to
        write the same graph. An event that reports how a task's run ended,
        the report of a run that did not finish among them, is answered when
        the revision re-planned that task past it (:func:`_answered`); one that
        reports tasks the orchestrator added, when the revision holds them all;
        every other event, a finished run's report among them, is kept. With the
        last report of the turn answered, no report is left unread.
        ``written`` is each task's status and attempt as the planner wrote
        them, before the orchestrator's own were carried over.
        """
        marks = self._aligned_marks()
        tasks = self.graph.by_id()
        kept, dropped = [], []
        for event, mark in zip(self.events, marks):
            answered = (mark.outcome is not None and _answered(mark.outcome, written, tasks)) or (
                bool(mark.added) and all(task_id in written for task_id in mark.added)
            )
            (dropped if answered else kept).append((event, mark))
        if not dropped:
            return
        self.events = [event for event, _ in kept]
        self._marks = [mark for _, mark in kept]
        if any(mark.report for _, mark in dropped) and not any(m.report for m in self._marks):
            self.unread_reports = False
        self._urgent = any(mark.urgent for mark in self._marks)
        if not self._loud():
            self._first_event_at = None
        self._save_events()
        self.s.log(
            f"[planner] revision {len(self.graph.revisions)} already answered "
            f"{len(dropped)} event(s) of its turn: " + "; ".join(event for event, _ in dropped)
        )

    def _only_background(self, running: dict) -> bool:
        """Whether every running task is one that runs for hours beside the graph (a campaign)."""
        if not running or self._settling is not None:
            return False
        return not self._foreground(running)

    def _loud(self) -> bool:
        """Whether an event waits that may wake a turn: any but a quiet one."""
        return any(not mark.quiet for mark in self._aligned_marks())

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
                    urgent=True,
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
                self._note(said, urgent=True, outcome=outcome)
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
                    urgent=bool(failed),
                    outcome=outcome,
                    quiet=not failed
                    and not unknown
                    and (task.kind == graph_mod.CHECK_KIND or self._answered_ahead(task)),
                )
            if task.status == "failed" and task.priority == "nice_to_have":
                task.status = "abandoned"
                self._note(f"nice_to_have task {task.id} abandoned", urgent=True, outcome=outcome)
            if task.status in ("failed", "abandoned"):
                self._note_soft_dependents(task)
            if task.kind == "survey" and task.status == "done":
                self._add_kernel_tasks(task)
            self._add_defect_tasks()
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
        self.s.checkpoint()
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
                urgent=True,
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
        devices on the check pool as every check does (:attr:`Services.admit`).
        The future holds what a worker's thread returns, :data:`CHECKED` and
        the keys that failed, once every check is in. Its delivery-tier checks
        wait for the delivery tier, as any task's do.
        """
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
        """The ready regression reviews to keep waiting; those with nothing to review close.

        Only while the run binds the cross-model rule (its own paths are
        known). One regression review runs at a time: while one runs, the
        others wait and what lands meanwhile joins the next batch. With none
        running, a review whose batch is empty closes with no turn
        (:meth:`_close_reviewed`), and of the rest the one on the longest
        chain starts once the batch's window has closed (:meth:`_batch_open`).
        """
        own = self.validation["own_paths"]
        if not own:
            return set()
        reviews = [
            t
            for t in graph_mod.ready(self.graph, self._receipts_landed)
            if graph_mod.is_regression_review(t)
        ]
        if not reviews:
            return set()
        held = {t.id for t in reviews}
        if any(
            t.status == "running" and graph_mod.is_regression_review(t) for t in self.graph.tasks
        ):
            return held
        batch = graph_mod.regression_batch(self.graph, own, self._landed_attempts())
        if not batch:
            for review in reviews:
                self._close_reviewed(review)
            return set()
        if self._batch_open(batch, own):
            return held
        path = graph_mod.critical_path_minutes(self.graph)
        first = max(reviews, key=lambda t: path.get(t.id, 0))  # the first of equals, in graph order
        return held - {first.id}

    def _batch_open(self, batch: Sequence[str], own: Sequence[str]) -> bool:
        """Whether more changes to shared code may still join ``batch`` before its review starts.

        While a task that changes shared code runs, for :attr:`regression_window_s`
        after the first of the batch landed. A change whose landing this run
        did not record counts as landed long ago.
        """
        if not any(
            t.status == "running" and graph_mod.changes_shared_code(t, own)
            for t in self.graph.tasks
        ):
            return False
        landed = [self._landings[tid][0] for tid in batch if tid in self._landings]
        return bool(landed) and time.time() - min(landed) < self.regression_window_s

    def _landed_attempts(self) -> dict[str, int]:
        """Per task whose commits landed, the attempt that landed them last."""
        return {tid: attempt for tid, (_, attempt) in self._landings.items()}

    def _record_batch(self, review: graph_mod.Task) -> None:
        """Record, as a regression review starts, the batch it covers (:attr:`~.graph.Task.covers`)."""
        own = self.validation["own_paths"]
        if not own or not graph_mod.is_regression_review(review):
            return
        tasks = self.graph.by_id()
        batch = graph_mod.regression_batch(self.graph, own, self._landed_attempts())
        review.covers = [graph_mod.landing(tasks[tid]) for tid in batch]
        self.s.log(
            f"[regression] {review.id} starts on the changes to shared code landed since the "
            f"last review: {', '.join(batch) or 'none'}"
        )

    def _close_reviewed(self, review: graph_mod.Task) -> None:
        """Close a regression review whose batch is empty: earlier ones covered everything.

        Every change to shared code it follows landed and was covered by a
        regression review that ran and passed; no agent turn runs for it.
        """
        tasks = self.graph.by_id()
        follows = {graph_mod.landing(tasks[d]) for d in review.depends_on if d in tasks}
        by = sorted(
            t.id
            for t in self.graph.tasks
            if graph_mod.is_regression_review(t)
            and t.status == "done"
            and set(t.covers or ()) & follows
        )
        review.status, review.covers = "done", []
        covered = f": {', '.join(by)} covered what it follows" if by else ""
        self._note(
            f"task {review.id} (review) closed with no turn: no change to shared code was left "
            f"for it to review{covered}",
            outcome=_Outcome(review.id, review.attempts, "done"),
        )
        self.revisions_since_progress = 0
        self._save_graph()
        self._save_events()
        self.s.checkpoint()
        self.s.finished(review)
        self._wake_soft_waiters()

    def _regression_standing(self) -> str:
        """Where the cross-model regression reviews stand, for the planner; "" without own paths."""
        own = self.validation["own_paths"]
        if not own:
            return ""
        batch = graph_mod.regression_batch(self.graph, own, self._landed_attempts())
        reviews = [t for t in self.graph.tasks if graph_mod.is_regression_review(t)]
        running = [t.id for t in reviews if t.status == "running"]
        waiting = [t.id for t in reviews if t.status == "pending"]
        said = []
        if batch:
            landed = [self._landings[tid][0] for tid in batch if tid in self._landings]
            minutes = (time.time() - min(landed)) / 60 if landed else None
            since = f", the first landed {minutes:.0f} min ago" if minutes is not None else ""
            said.append(
                f"Landed changes to shared code no regression review covers yet, the batch the "
                f"next one takes: {', '.join(batch)}{since}."
            )
        else:
            said.append("Every landed change to shared code is covered by a regression review.")
        if running:
            said.append(f"Running: {', '.join(running)}; the others wait for it.")
        if waiting:
            said.append(
                f"Not started: {', '.join(waiting)}. A regression review starts once its "
                f"dependencies are done and no other runs; while a task that changes shared code "
                f"runs, it waits for that change up to {self.regression_window_s / 60:.0f} min "
                f"after its batch's first landing. One whose batch is empty closes with no turn."
            )
        return " ".join(said)

    def _unreviewed_landings(self) -> list[str]:
        """The landed changes to shared code no regression review covers (none without own paths)."""
        own = self.validation["own_paths"]
        if not own:
            return []
        return graph_mod.regression_batch(self.graph, own, self._landed_attempts())

    def _add_kernel_tasks(self, survey: graph_mod.Task) -> None:
        """One kernel task per queue row of the vocabulary the survey wrote, with no planner turn.

        Each smiths its row (``ref``) and touches the row's entry and its test
        (:func:`~..onboard.vocabulary.entry_paths`), an entry of the target's
        own ``ops/`` under the directory the survey names (its ``ref``), else
        the run's own paths. A row that depends on others waits for their
        tasks, and a row a kernel task already names is left to that task. They
        start as soon as devices allow; the planner hears what was added, wires
        the rest of the graph to them and may reshape those not yet running.

        Rows of the core unit are wired at once into the core assembly that has
        not started (:meth:`_wire_core_rows`), so they rank on its critical path
        and start before the rows of later units.
        """
        try:
            vocabulary = vocab_mod.load_vocabulary(self.workspace / vocab_mod.VOCABULARY_FILENAME)
        except vocab_mod.VocabularyError as exc:
            self.s.log(f"[survey] no kernel tasks added: {exc}")
            return
        own = self.validation["own_paths"]
        target = survey.ref or (own[0] if own else "")
        kernels = {t.ref: t.id for t in self.graph.tasks if t.kind == "kernel"}
        taken = {t.id for t in self.graph.tasks}
        units = {c.unit for c in self.contract.checks}
        required = {c.unit for c in self.contract.required()}
        rows = [item for item in vocabulary.queue if str(item.n) not in kernels]
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
                    # A verify row's smith runs tests whose jobs take their own devices.
                    gpus=0 if item.mode == vocab_mod.MODE_VERIFY else 1,
                    est_minutes=120 if item.writes_implementation else 60,
                    priority=("nice_to_have" if item.unit in units - required else "required"),
                    ref=str(item.n),
                )
            )
        if not added and not unplaced:
            return
        by_row = {item.n: item for item in vocabulary.queue}
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
        said = [f"task {survey.id} wrote {len(vocabulary.queue)} vocabulary queue row(s)"]
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
            said.append(
                f"{', '.join(shared)} change code other models run: add them to the `depends_on` "
                f"of the regression review that has not started, or add one"
            )
        # What asks the planner for nothing is answered by a revision that holds the tasks.
        self._note("; ".join(said), added=() if unplaced or shared else tuple(t.id for t in added))

    def _wire_review(self, task: graph_mod.Task) -> None:
        """Put ``task``, added by the orchestrator, before the regression review not yet started.

        A task that changes shared code needs a review after it, or the graph
        fails its own validation and the next planner turn is spent wiring it.
        """
        if not graph_mod.changes_shared_code(task, self.validation["own_paths"]):
            return
        for review in self.graph.tasks:
            if (
                graph_mod.is_regression_review(review)
                and review.status == "pending"
                and task.id not in review.depends_on
            ):
                review.depends_on.append(task.id)

    def _add_delivery_task(self) -> None:
        """Add the task that writes the delivery files, when the graph has none.

        Full support includes the files on disk (:meth:`_support_gaps`); that
        a graph has a task for them is the orchestrator's to see to, not the
        planner's to remember.
        """
        if self.s.delivery_task is None:
            return
        spec = self.s.delivery_task(self.graph)
        if not spec:
            return
        task = graph_mod.Task.from_dict(dict(spec))
        self.graph.tasks.append(task)
        self._wire_review(task)
        self.graph.revisions.append(
            graph_mod.Revision(
                len(self.graph.revisions) + 1,
                f"orchestrator: {task.id} writes the delivery files",
            )
        )
        self._note(
            f"the orchestrator added {task.id} ({task.kind}): it writes the delivery files "
            f"(the CI accuracy test, its test list entry, the accuracy reference) once "
            f"{', '.join(task.depends_on) or 'it can'} is done",
            added=(task.id,),
        )
        self._save_graph()

    def _add_defect_tasks(self) -> None:
        """Add the task that fixes each engine defect a worker reported, with no planner turn.

        A smith's failed receipt can name a defect in code outside its entry
        (``engine_defect``); the stage registers it and proposes the fix task.
        One whose touches the run may not change is left to the planner, who
        hears of it either way and may reshape it.
        """
        if self.s.defect_tasks is None:
            return
        added: list[graph_mod.Task] = []
        for spec in self.s.defect_tasks(self.graph):
            task = graph_mod.Task.from_dict(dict(spec))
            if any(t.id == task.id for t in self.graph.tasks) or not all(
                graph_mod.inside(p, self.validation["allowed_paths"]) for p in task.touches
            ):
                self._note(
                    f"an engine-defect fix ({task.id}, touching {', '.join(task.touches)}) is not "
                    f"added: the run may not change those paths; plan its fix or workaround",
                    urgent=True,
                )
                continue
            self.graph.tasks.append(task)
            self._wire_review(task)
            added.append(task)
        if not added:
            return
        names = ", ".join(t.id for t in added)
        self.graph.revisions.append(
            graph_mod.Revision(
                len(self.graph.revisions) + 1,
                f"orchestrator: {names} fix the engine defects workers reported",
            )
        )
        if self._planner is not None:
            self._added_in_turn |= {t.id for t in added}
        self._note(
            f"the orchestrator added {names}: each fixes an engine defect a smith's failed "
            f"receipt reported (its goal says which and how); retry the failed task once the "
            f"fix lands, add the regression review its shared touches need, or reshape it",
            urgent=True,
            added=tuple(t.id for t in added),
        )
        self._save_graph()

    def _promote(self) -> str:
        """Add a promote task when one is due; what keeps the target from delivery, or "".

        Kernel tasks on a shared entry land local variants in the target so they run side
        by side (``onboard/promotion.py``). Entry by entry, once its variants have landed
        and the target imports them, a kernel task merges them into the catalog, beside
        the feature work. A check task that has not started waits for it, so it
        measures the merged entries once rather than the variants and then again.
        While variants wait the target is not delivered, and the planner hears why
        once per reason.
        """
        if self.s.promotion is None:
            return ""
        promotion = self.s.promotion(self.graph)
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
            f"target's local variants into the shared catalog, so a cross-model regression "
            f"review must follow it",
            added=(task.id,),
        )
        self._save_graph()
        return f"{task.id} promotes them"

    def _timeout(self) -> float | None:
        """How long to wait for something in flight to end before looking again (None: until then)."""
        timeout = None
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
        end on their own, their outcomes never read. Or never return: a
        reload that drained replaces the process.
        """
        self._mark_reloaded()
        if self.s.reload is not None:
            self._watch = drain_mod.RequestWatch(self.workspace, poll_s=self.poll_s)
            self._watch.start()
        checks = ThreadPoolExecutor(
            max_workers=self.max_parallel_checks, thread_name_prefix="mv2-check"
        )
        tasks = ThreadPoolExecutor(
            max_workers=max(1, self.max_parallel), thread_name_prefix="mv2-task"
        )
        planner = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mv2-planner")
        critic = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mv2-critic")
        self._check_pool = checks
        verdict = None
        crashed = False
        try:
            try:
                verdict = self._drive(tasks, planner, critic)
                return verdict
            except Exception:
                # The loop's own failure: what finished is recorded before the process
                # goes, and nothing still running is waited for; a resumed run starts it again.
                crashed = True
                self._record_finished()
                raise
            finally:
                # Nothing re-executes from here: what waits at the door goes on,
                # so the pools below can finish it.
                if self._drain is not None:
                    self._end_drain(reload_mod.PHASE_ABANDONED)
                self.inflight.open()
                # A task thread waiting on its soft dependencies would wait for ever.
                with self._ended:
                    self._closing = True
                    self._ended.notify_all()
                stopped = self._stop.done() or crashed
                for pool in (critic, planner, tasks):
                    pool.shutdown(wait=not stopped, cancel_futures=stopped)
        finally:
            if self._watch is not None:
                self._watch.stop()
            # After the tasks: a task still running submits its checks here.
            stopped = self._stop.done() or crashed
            checks.shutdown(wait=not stopped, cancel_futures=stopped)
            self._check_pool = None
            if verdict is not None:
                self.inflight.stop(f"closed:{verdict}")

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

    def _drive(
        self, tasks: ThreadPoolExecutor, planner: ThreadPoolExecutor, critic: ThreadPoolExecutor
    ) -> str:
        running: dict[str, Future] = {}
        self._running = running
        self._reload_check()
        # A resumed run whose contract held before it stopped keeps a milestone for it.
        self._correctness(*self._evaluated())
        if not self.graph.tasks and self._drain is None and not self._stop.done():
            self._wake_planner(planner)
        while True:
            if self._stop.done():
                self.s.log(f"[stop] {self._stop.result()}: nothing more starts")
                return f"stopped: {self._stop.result()}"
            # Taken before the request is read, so a change after the read still wakes the wait.
            signal = self._watch.signal() if self._watch is not None else None
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
            if self.s.critique is not None:
                self._critic_step(critic)
            self._reload_check()
            if self._drain is not None:
                changed = self._drain_step(running)
                if self._drain is not None:
                    if changed is not None:
                        self._wait_while_draining(running, [signal, changed])
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
            # One agent slot stays free for the planner (agent_gate.py).
            slots = max(AgentGate(self.max_parallel).task_slots - workers, 0)
            self._resumed &= {t.id for t in self.graph.tasks if t.status == "pending"}
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
            idle = (
                not running
                and self._settling is None
                and self._planner is None
                and self._critic is None
            )
            # Nothing in flight but a campaign: what the graph owes is read as
            # when idle, for the campaign gives way once full support holds.
            spent = idle or (
                not self._foreground(running)
                and self._settling is None
                and self._planner is None
                and self._critic is None
            )
            if spent:
                marker, evaluation = self._evaluated()
                self._correctness(marker, evaluation)
                holds = evaluation.holds and not self._required_open(campaign=False)
                unpromoted = self._promote()
                if holds and unpromoted:
                    if unpromoted != self._unpromoted_said:
                        self._unpromoted_said = unpromoted
                        self._note(
                            f"the contract holds, but local variants wait for promotion: "
                            f"{unpromoted}",
                            urgent=True,
                        )
                    holds = False
                unreviewed = self._unreviewed_landings() if holds else []
                gaps = self._support_gaps(marker, evaluation) if holds else []
                # What only the delivery checks hold back is the delivery phase's to report.
                self._files_alone = bool(gaps) and all(
                    g.startswith(DELIVERY_FILES_GAP) for g in gaps
                )
                if holds and not self.unread_reports and not gaps:
                    if idle:
                        return "delivered"
                    self._stop_campaigns(running)
                if unreviewed:
                    self.revisions_since_progress += 1
                    self._note(
                        "every required check holds, but no cross-model regression review "
                        f"covers {', '.join(unreviewed)}, which landed changes to code other "
                        "models run: add a required `review` task with `regression: true`; the "
                        "run is not delivered before one has passed",
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
                    if blocked:
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
                        return "delivered" if self._files_alone else "planner-stalled"
                    self._wake_planner(planner)
            in_flight = self._in_flight(running) + ([signal] if signal is not None else [])
            wait(in_flight, timeout=self._timeout(), return_when=FIRST_COMPLETED)

    def _in_flight(self, running: dict[str, Future]) -> list[Future]:
        """The tasks, re-checks, planner turn and critic read in flight, as futures to wait on.

        A task, planner turn or critic read that ended since the loop looked
        ends the wait at once, and is recorded. A batch of re-checks is read
        once all of it is in, so until then only its members still out are
        waited on: one already in would end every wait at once. A stop ends
        the wait too.
        """
        in_flight = [*running.values(), self._stop]
        if self._settling is not None:
            batch = [f for _, f in self._settling[1]]
            in_flight += [f for f in batch if not f.done()] or batch
        if self._planner is not None:
            in_flight.append(self._planner)
        if self._critic is not None:
            in_flight.append(self._critic)
        return in_flight

    # ----------------------------------------------------------------- reload
    def _mark_reloaded(self) -> None:
        """The loop runs again after a reload re-executed the process: the reload is complete."""
        record = reload_mod.read_record(self.workspace)
        if record and record.get("phase") == reload_mod.PHASE_REEXEC:
            record.update(
                phase=reload_mod.PHASE_RELOADED, reloaded_at=time.time(), reloaded_pid=os.getpid()
            )
            self._write_reload_record(record)
            self.s.log("[reload] " + reload_mod.describe(record))

    def _write_reload_record(self, record: Mapping[str, Any]) -> None:
        try:
            reload_mod.write_record(self.workspace, record)
        except OSError as exc:
            self.s.log(f"[reload] could not write {reload_mod.RECORD_FILENAME}: {exc}")

    def _drain_waits(self) -> list[str]:
        """What the drain waits for: the work in flight, a planner turn and a critic read."""
        waits = [f"{h['kind']} {h['what']}" for h in self.inflight.held()]
        if self._planner is not None:
            waits.append("planner turn")
        if self._critic is not None:
            waits.append("critic read")
        return waits

    def _drain_record(self, phase: str, now: float, **extra: Any) -> dict:
        drain = self._drain or {}
        waited = [{"what": what, "s": round(s, 1)} for what, s in drain.get("ended", {}).items()]
        return {
            "id": drain.get("id"),
            "phase": phase,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "requested_at": drain.get("requested_at"),
            "by": drain.get("by"),
            "since": drain.get("since"),
            "deadline": drain.get("deadline"),
            "force": drain.get("force"),
            "overdue": drain.get("overdue"),
            "waiting_for": drain.get("waiting", []),
            "waited": waited,
            "drained_s": round(now - float(drain.get("since") or now), 1),
            "updated_at": now,
            **extra,
        }

    def _reload_check(self) -> None:
        """Start a drain on a reload request, end it on a withdrawn one, follow one that changed."""
        if self.s.reload is None:
            return
        request = reload_mod.read_request(self.workspace)
        now = time.time()
        if self._drain is None:
            if request is None or _asked(request) == self._failed_request:
                return
            timeout = float(request.get("timeout_s", reload_mod.DEFAULT_DRAIN_TIMEOUT_S))
            self.inflight.close()
            self._drain = {
                "id": request.get("id"),
                "requested_at": request.get("requested_at"),
                "by": request.get("by"),
                "since": now,
                "deadline": now + timeout,
                "force": bool(request.get("force")),
                "overdue": False,
                "waiting": self._drain_waits(),
                "ended": {},
                "asked": _asked(request),
            }
            self._write_reload_record(self._drain_record(reload_mod.PHASE_DRAINING, now))
            self.s.log(
                f"[reload] requested by {request.get('by')}: draining. No task, integration, "
                f"check or planner turn starts until the process re-executes; waiting for "
                f"{', '.join(self._drain['waiting']) or 'nothing'}"
            )
            return
        if request is None:
            self._end_drain(reload_mod.PHASE_CANCELLED)
            self.s.log("[reload] the request was withdrawn; the run goes on")
            return
        deadline = float(self._drain["since"]) + float(
            request.get("timeout_s", reload_mod.DEFAULT_DRAIN_TIMEOUT_S)
        )
        force = bool(request.get("force"))
        self._drain["asked"] = _asked(request)
        if (deadline, force) != (self._drain["deadline"], self._drain["force"]):
            overdue = self._drain["overdue"] and now >= deadline
            self._drain.update(deadline=deadline, force=force, overdue=overdue)
            self._write_reload_record(self._drain_record(reload_mod.PHASE_DRAINING, now))

    def _drain_step(self, running: dict[str, Future]) -> Future | None:
        """Note what ended, report a drain past its deadline, re-execute once it is drained.

        Returns the future the next change to what is in flight completes, or
        None when an outcome is in for the loop to record first; while it
        re-executes the process, it does not return.
        """
        drain = self._drain
        changed = self.inflight.changed()
        waits = self._drain_waits()
        now = time.time()
        for what in drain["waiting"]:
            if what not in waits:
                drain["ended"].setdefault(what, now - drain["since"])
        if waits != drain["waiting"]:
            drain["waiting"] = waits
            self._write_reload_record(self._drain_record(reload_mod.PHASE_DRAINING, now))
        if not waits:
            # A check that just ended hands its verdict to its task's thread, which
            # returns at once: the task's outcome is recorded here, not left behind.
            pending = self._in_flight(running)
            if (
                self._outcome_in(running)
                or wait(pending, timeout=DRAIN_SETTLE_S, return_when=FIRST_COMPLETED).done
            ):
                return None
            self._replace([])
        elif drain["force"] and now >= drain["deadline"]:
            self._replace(waits)
        elif now >= drain["deadline"] and not drain["overdue"]:
            drain["overdue"] = True
            self._write_reload_record(self._drain_record(reload_mod.PHASE_DRAINING, now))
            self.s.log(
                f"[reload] past its deadline after {now - drain['since']:.0f}s, still waiting for "
                f"{', '.join(waits)}; it keeps waiting. `driver reload --force` re-executes "
                f"without them, `driver reload --cancel` withdraws the request"
            )
        return changed

    def _outcome_in(self, running: dict[str, Future]) -> bool:
        """Something the loop records has ended since it last looked."""
        settled = self._settling is not None and all(f.done() for _, f in self._settling[1])
        planned = self._planner is not None and self._planner.done()
        read = self._critic is not None and self._critic.done()
        return settled or planned or read or any(f.done() for f in running.values())

    def _wait_while_draining(
        self, running: dict[str, Future], wake: Sequence[Future | None]
    ) -> None:
        drain = self._drain
        timeout = None
        if drain["force"] or not drain["overdue"]:
            timeout = max(float(drain["deadline"]) - time.time(), 0.0)
        in_flight = self._in_flight(running) + [f for f in wake if f is not None]
        wait(in_flight, timeout=timeout, return_when=FIRST_COMPLETED)

    def _end_drain(self, phase: str, **extra: Any) -> None:
        """The drain is over without a re-exec: the door opens and the run goes on."""
        record = self._drain_record(phase, time.time(), ended_at=time.time(), **extra)
        self._drain = None
        self.inflight.open()
        reload_mod.remove_request(self.workspace)
        self._write_reload_record(record)

    def _replace(self, left: list[str]) -> None:
        """Checkpoint and hand over to :attr:`Services.reload`; returns only if it did not replace us.

        ``left`` is what a forced reload cuts short.
        """
        now = time.time()
        record = self._drain_record(
            reload_mod.PHASE_REEXEC, now, reexec_at=now, left=list(left), forced=bool(left)
        )
        self._save_graph()
        self._save_events()
        self.s.checkpoint()
        self.inflight.stop("reload")
        self._write_reload_record(record)
        reload_mod.remove_request(self.workspace)
        # Handed over: from here the record speaks for this drain.
        drain, self._drain = self._drain, None
        try:
            self.s.reload(record)
            error = "the re-exec returned without replacing the process"
        except OSError as exc:
            error = f"{type(exc).__name__}: {exc}"
        self.inflight.stop(None)
        self._drain = drain
        self._failed_request = drain["asked"]
        self._end_drain(reload_mod.PHASE_FAILED, error=error, reexec_at=now)
        self.s.log(
            f"[reload] {error}; the run goes on in this process, on the code it started with"
        )


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
        _write_whole(events, json.dumps(saved, indent=1))
    graph = graph_mod.Graph(
        tasks=list(tasks), revisions=[graph_mod.Revision(1, f"orchestrator: {reason}")]
    )
    partial = path.with_name(path.name + ".partial")
    graph.save(partial)
    os.replace(partial, path)
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
    said = f"task {task.id} (check) ran its checks with no agent turn"
    if not failed:
        return f"{said}; they passed"
    return (
        f"{said}; checks failed: {', '.join(failed)}. No agent has looked: an `investigate` "
        "task finds why, or retry it (`status: pending`) once what it measures has changed"
    )


def _asked(request: Mapping[str, Any]) -> tuple:
    """A reload request as one ask: a later ``driver reload`` makes it another."""
    return request.get("id"), request.get("updated_at")


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


def _reshaped(before: graph_mod.Graph, after: graph_mod.Graph) -> bool:
    """The first graph, or a revision that drops or adds more than half of the open tasks."""
    if not before.tasks:
        return True
    old = {t.id for t in before.tasks if t.status in ("pending", "running")}
    new = {t.id for t in after.tasks if t.status in ("pending", "running")}
    changed = len(old ^ new)
    return changed > max(len(old), 1) / 2


__all__ = [
    "CEILING",
    "CHECKED",
    "DEFAULT_MAX_PARALLEL_CHECKS",
    "DEFAULT_MAX_PARALLEL_TASKS",
    "DONE",
    "ESCALATE",
    "FINISHED",
    "GRAPH_UPDATED",
    "GraphOrchestrator",
    "PlannerAnswer",
    "PlannerTurn",
    "Services",
    "seed",
    "stoppable",
]
