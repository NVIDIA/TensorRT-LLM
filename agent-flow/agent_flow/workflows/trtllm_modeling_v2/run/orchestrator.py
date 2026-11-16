# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The graph-driven run loop.

Code writes the task graph: the survey first, then one task per row the survey
wrote and the tasks of each deployment the spec states (:meth:`GraphOrchestrator._expand`).
No agent plans. This loop starts tasks as agent turns and paths allow, runs each
finished task's checks itself, and applies what the outer agent session resolves.
A task that cannot go on is ``blocked`` and asks the outer session
(``run/escalation.py``); everything that does not depend on it goes on.

Nothing is verified a second time in the middle of a run: a result in the ledger
never starts work. When every task is done or skipped, the target is frozen and
one final pass runs on that candidate (:meth:`GraphOrchestrator._final_pass`):
every requested check must have a row whose key is the candidate's, so a check
whose key is unchanged is not measured again and one whose key changed is. The
run ends ``delivered`` when every requested check passed there, ``partial`` when
a task was skipped or a check could not be verified, and ``escalated`` when
something is blocked, a task or the final pass itself.

Each task's worker runs in a thread of its own and every check on one bounded
pool. Only the loop's own thread changes the graph or writes it: a resolution
command is applied here, on the loop's next pass, and is applied once. There is
one counter of agent turns (``max_parallel``); devices are the pool's, and the
spool is their only ledger.

The workflow's services arrive as callables (:class:`Services`), so the loop
runs under fakes on CPU exactly as it runs under the workflow.
"""

from __future__ import annotations

import contextlib
import errno
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from ..agents import decisions as decision_table
from ..agents.turns import TurnEnded
from ..checks import delivery as delivery_rules
from ..infra.compute import controller as controller_mod
from ..operators import coverage as vocab_mod
from ..operators import rows as rows_mod
from . import escalation as escalation_mod
from . import evidence as evidence_mod
from . import graph as graph_mod
from .validation import contract as contract_mod
from .validation.completion import Completion, assess
from .workspace.integration import TaskNotStarted

#: What a worker's receipt says (``agents/decisions.py``).
READY, BLOCKED = decision_table.READY, decision_table.BLOCKED
#: How many tasks run at once, and how many checks, unless the caller says otherwise.
DEFAULT_MAX_PARALLEL_TASKS = 9
DEFAULT_MAX_PARALLEL_CHECKS = 8
#: A state write that fails with one of these waits for room instead of ending the run.
STORAGE_FULL = frozenset({errno.EDQUOT, errno.ENOSPC})
#: How the skeleton service's answer starts when every directory on the way exists:
#: nothing was committed, and nothing is journaled for it.
NOTHING_OWED = "nothing to commit"
#: The two forms a deployment takes: one engine, or a context side and a generation side.
AGGREGATED, DISAGGREGATED = "aggregated", "pd"
#: The longest a task keeps of how its previous attempt ended.
LAST_CHARS = 900
#: Seconds a controller stays once everything left waits for an answer, before it leaves
#: with ``escalated``: one look of the outer session (it looks every five to nine
#: minutes) and the time to read a receipt and answer.
ANSWER_WAIT_S = 900.0
#: Per workspace, the stop a run this process drives there answers (:func:`stoppable`).
_STOPS: dict[Path, Future] = {}
_STOPS_LOCK = threading.Lock()


@dataclass(frozen=True)
class Deployment:
    """One deployment of the spec, as the task list needs it (``tasks.deployments_of``)."""

    name: str
    kind: str = AGGREGATED
    #: The contract checks its session owes.
    checks: tuple[str, ...] = ()
    #: The deployment whose own score this one is judged against; empty for none.
    relative_to: str = ""
    #: Its session's job line: devices on one node, or whole nodes past one, and its wall.
    gpus: int = 0
    nodes: int = 0
    wall: int = 0
    #: The devices the own test of one of its sides may hold: its ranks, one node's at most.
    row_gpus: int = 1


@dataclass
class Services:
    """The workflow, as the loop needs it."""

    #: One worker for a task; returns what its receipt says, ``ready`` or ``blocked``
    #: (blocking; runs in a thread). Lines after the first are a note kept with the task.
    work: Callable[[graph_mod.Task], str]
    #: Run one contract check now and return its gate result (``name``, ``member``, ``passed``)
    #: (blocking; several run at once, each in a thread).
    check: Callable[[contract_mod.Check], Any]
    checkpoint: Callable[[], None] = lambda: None
    log: Callable[[str], None] = lambda message: None
    #: The key of a check's evidence on a commit of the target (``""``: the target as it
    #: is): what the check measures there, by content. Two states of the target with the
    #: same key hold the same evidence; ``""`` is "not established", and nothing is
    #: reused for it.
    key: Callable[[contract_mod.Check, str], str] = lambda check, commit="": ""
    #: The loop has recorded how a task's attempt ended: its answer is consumed.
    finished: Callable[[graph_mod.Task], None] = lambda task: None
    #: Freeze the candidate of the final pass: pin the target's head under the landing
    #: lock, and return its commit. ``None``: the target as it is, unnamed.
    freeze: Callable[[], str] | None = None
    #: The final tests on the frozen candidate (its commit): one selection of test files,
    #: run once as jobs that state their devices, each failing case compared with the
    #: base (blocking; in the final pass's thread). What they found
    #: (``regression.TestsResult``), or ``None`` with nothing to run. ``None``: no final
    #: tests in this run.
    final_tests: Callable[[str], Any] | None = None
    #: The delivery checks on disk (the CI accuracy test, the test list entry, the
    #: accuracy reference, the target's files and commits), as cheap as reading files:
    #: what fails, one line each; empty when all pass. ``None``: none are checked here.
    delivery_files: Callable[[], list[str]] | None = None
    #: End the agent turn in flight for a task, as its own limit would; whether one was.
    end_turn: Callable[[graph_mod.Task], bool] = lambda task: False
    #: Why no job that is ready can start where the run runs before that place ends (one
    #: host inside a job: the pool's ``late``); ``""`` while one can.
    late: Callable[[], str] = lambda: ""
    #: The spec's deployments in its order, the first first.
    deployments: Callable[[], Sequence[Deployment]] = lambda: ()
    #: The files the delivery requires for CI beside the target: its accuracy test, the
    #: directory of the test lists, the reference file of its anchor's row. Empty when
    #: the run owes none.
    ci_files: Callable[[], Sequence[str]] = lambda: ()
    #: Commit the package skeleton on the way to these files, before the tasks that write
    #: them start: the directories and their shared initializers. What it committed.
    skeleton: Callable[[Sequence[str]], str] | None = None
    #: The first deployment's session passed: pin the ``onboarded`` ref, once.
    onboarded: Callable[[graph_mod.Task], Any] | None = None
    #: The receipt a task's latest attempt left: its question when it blocks; empty for none.
    receipt: Callable[[graph_mod.Task], Mapping[str, Any]] = lambda task: {}
    #: What an answer of the outer session reaches outside the graph: the spec, the pool, anchors.
    hooks: Callable[[], escalation_mod.Hooks] | None = None


#: What a ``check`` task ends with once its checks are in; no worker writes it.
CHECKED = "CHECKED"
#: What means a task's attempt did what it was asked: the receipt's ``ready``, and the one
#: the run writes itself.
FINISHED = frozenset({READY, CHECKED})


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


def _waves(checks: Sequence[contract_mod.Check]) -> list[list[contract_mod.Check]]:
    """``checks`` in the waves they are measured in: each after the one it is judged against.

    A check that names none, or one that is not among these, is in the first wave; one
    judged against another's result is in the wave after that one's, so a chain of three
    is three waves and never two checks of it at once. Names that go round in a circle
    (the contract compiles none) end in one last wave together.
    """
    by_key = {c.key: c for c in checks}
    level: dict[str, int] = {}
    left = list(checks)
    while left:
        placed = [c for c in left if c.after not in by_key or c.after in level]
        if not placed:
            top = max(level.values(), default=-1) + 1
            level.update({c.key: top for c in left})
            break
        level.update({c.key: level.get(c.after, -1) + 1 for c in placed})
        left = [c for c in left if c.key not in level]
    top = max(level.values(), default=-1)
    return [[c for c in checks if level[c.key] == n] for n in range(top + 1)]


@dataclass(frozen=True)
class FinalRow:
    """One requested check in the final pass."""

    #: Its key on the candidate; ``""`` when it has no content identity there.
    key: str
    #: The row it has under that key; ``None``: it has none.
    row: contract_mod.Verdict | None
    #: Whether this pass measured it; ``False``: a row already had the candidate's key.
    measured: bool


@dataclass
class FinalPass:
    """What the one pass on a frozen candidate found (:meth:`GraphOrchestrator._final_pass`)."""

    #: The commit that was frozen; ``""`` when the target names none.
    candidate: str
    #: The contract the pass enumerated.
    contract: contract_mod.Contract
    #: Each requested check it looked at, by the check's name, in the contract's order.
    rows: dict[str, FinalRow] = field(default_factory=dict)
    #: The requested checks it did not run, each with why: its task is not done.
    waived: dict[str, str] = field(default_factory=dict)
    #: What the final tests found (``regression.TestsResult``); ``None`` when none ran,
    #: and ``tests_said`` says why. ``tests_lost``: they were owed and left no result.
    tests: Any = None
    tests_said: str = ""
    tests_lost: bool = False
    #: What of the target moved while the pass ran, read once when it ended; ``""``: nothing.
    moved: str = ""

    def evaluation(self) -> contract_mod.Evaluation:
        """The requested checks it looked at, each by its row on the candidate."""
        return contract_mod.judged(
            self.contract, {name: found.row for name, found in self.rows.items()}
        )

    def as_record(self, verdict: str) -> dict[str, Any]:
        """What the closing report says of the pass: each requested check, and the final tests."""
        outcomes = {}
        evaluation = self.evaluation()
        outcomes.update({c.key: "passed" for c in evaluation.passed})
        outcomes.update({c.key: "failed" for c, _ in evaluation.failed})
        outcomes.update({c.key: "not verified" for c, _ in evaluation.unverified})
        checks = []
        for check in self.contract.required():
            found = self.rows.get(check.key)
            if found is None:
                why = self.waived.get(check.key, "it is not one the pass looked at")
                checks.append({"check": check.key, "outcome": "not run", "key": "", "how": why})
                continue
            row = found.row
            if row is None:
                how = "it left no result on the candidate"
            elif found.measured:
                how = "measured in the final pass"
            else:
                how = "not measured again: a row had the candidate's key"
            said = "" if row is None else str(row.unverified or row.detail or "")
            checks.append(
                {
                    "check": check.key,
                    "outcome": outcomes.get(check.key, "not run"),
                    "key": found.key,
                    "how": how,
                    "said": said[:300],
                }
            )
        tests = None
        if self.tests is not None:
            # The tests' own result (`regression.TestsResult`), with its two readings.
            tests = {
                **asdict(self.tests),
                "passed": self.tests.passed,
                "summary": self.tests.summary(),
            }
        return {
            "candidate": self.candidate,
            "verdict": verdict,
            "checks": checks,
            "tests": tests,
            "tests_said": self.tests_said,
        }


#: What the loop asks when it blocks a task itself and neither the worker's receipt nor the
#: caller asks: by the kind of block, with the verbs that answer it. A whole run's escalation
#: for a used budget asked nothing, and its reader had to work out alone what could be done.
ASKS: dict[str, str] = {
    "attempts_exhausted": (
        "`{task}` used its {budget} attempt(s), and the last ended as was observed. Which "
        "is wrong: its work (`resolve retry {task} --note <what to do differently>` gives "
        "it one more attempt), what it is held to (`resolve change-requirement {task}`), "
        "or that the run owes it (`resolve skip {task}`)?"
    ),
    "check_failed": (
        "the check of `{task}` did not pass, and no task it follows has an attempt left. "
        "Whose is it: a task's (`resolve retry {task} --task <that task> --note <what to "
        "repair>` sends that one back, and the check runs again after it), the "
        "requirement's (`resolve change-requirement {task}`), or not owed (`resolve skip "
        "{task}`)?"
    ),
    "write_scope": (
        "may the run commit what `{task}` has to write? `resolve allow-shared {task} --path "
        "<prefix>` lets it in; `resolve skip {task}` leaves it out, and the run reports it "
        "not done"
    ),
    "skeleton_failed": (
        "the checkout did not take the commit `{task}` needs before it starts. Once it can "
        "(what was observed says what kept it), `resolve retry {task} --note <what was put "
        "right>` starts the task; `resolve edit-env {task} --set key=value` when a key of "
        "the environment is what has to change"
    ),
    "tree_not_prepared": (
        "no tree could be made for `{task}`, and no attempt was spent: it is the "
        "environment's (what was observed says what). Once it is put right, `resolve retry "
        "{task} --note <what was put right>` starts the task; `resolve edit-env {task} --set "
        "key=value` when a key of the environment is what has to change"
    ),
}


class GraphOrchestrator:
    def __init__(
        self,
        services: Services,
        contract: contract_mod.Contract,
        status: contract_mod.ContractStatus,
        workspace: Path,
        *,
        max_attempts: int = graph_mod.MAX_ATTEMPTS,
        max_parallel: int = DEFAULT_MAX_PARALLEL_TASKS,
        max_parallel_checks: int = DEFAULT_MAX_PARALLEL_CHECKS,
        poll_s: float = 5.0,
        allowed_paths: Iterable[str] = (),
        own_paths: Iterable[str] = (),
        repo_root: str | Path = "",
        baseline: str = "",
        answer_wait_s: float | None = None,
    ) -> None:
        self.s = services
        self.contract = contract
        #: State writes a full filesystem refused, by what they write: each is tried
        #: again as the loop goes round, and nothing new starts until all have landed.
        self._unwritten: dict[str, Callable[[], None]] = {}
        self.status = status
        self.graph_path = Path(workspace) / graph_mod.GRAPH_FILENAME
        self.graph = (
            graph_mod.Graph.load(self.graph_path) if self.graph_path.exists() else graph_mod.Graph()
        )
        self.workspace = Path(workspace)
        # A task recorded as running belonged to a process that is gone. What that process
        # left on disk is read before anything is dispatched: a task whose worker wrote a
        # blocked receipt for this attempt, and whose graph save was lost, is blocked as
        # its receipt says and is not run again. Any other starts again as the attempt it
        # was (a turn the controller's limit cut spends none), and its worker resumes from
        # what it left.
        restarted, reconciled = [], False
        for task in self.graph.tasks:
            if task.status != "running":
                continue
            try:
                left = dict(services.receipt(task) or {})
            except Exception:  # an unreadable receipt says nothing: the attempt runs again
                left = {}
            if left.get("decision") == BLOCKED and int(left.get("attempt") or 0) == task.attempts:
                self._block(
                    task,
                    escalation_mod.WORKER,
                    "blocked",
                    str(left.get("observed") or "its worker's receipt says blocked"),
                )
                reconciled = True
            else:
                task.status = "pending"
                restarted.append(task)
        #: Tasks that were running when the last process stopped and have not
        #: started again: they start ahead of every other ready task.
        self._resumed = {t.id for t in restarted}
        if restarted or reconciled:
            self._save_graph()
        if restarted:
            services.log(
                "[resume] the tasks running when the last process stopped start again: "
                + ", ".join(t.id for t in restarted)
            )
        #: How many attempts a task made by this run may use (``Task.budget``).
        self.max_attempts = max(1, max_attempts)
        #: The one counter of agent turns: how many run at once.
        self.max_parallel = max(1, max_parallel)
        #: Checks running at once, across every task and the re-checks of an idle run.
        self.max_parallel_checks = max(1, max_parallel_checks)
        #: How long the loop waits between looks for a resolution or a pool event.
        self.poll_s = poll_s
        self._status_lock = threading.Lock()
        #: One lock per check key: the same check never runs twice at once.
        self._check_locks: dict[str, threading.RLock] = {}
        self._check_locks_guard = threading.Lock()
        #: The checkout a kernel task's files are named from: a multi-rank entry's op matrix.
        self.repo_root = str(repo_root)
        #: The onboarded ref the spec names: its rows are the delta, and no deployment waits
        #: for a first one.
        self.baseline = str(baseline or "")
        self.validation = {
            "required_checks": [c.key for c in contract.required()],
            "allowed_paths": tuple(allowed_paths),
            "own_paths": tuple(own_paths),
            "known_checks": [c.key for c in contract.checks],
        }
        #: The task threads in flight, by task id, as the loop drives them.
        self._running: dict[str, Future] = {}
        #: Tasks whose work finished but some of whose checks raised: why, until recorded.
        self._inconclusive: dict[str, str] = {}
        #: Running tasks whose turn is over and whose checks run: they hold no agent turn.
        self._checking: set[str] = set()
        #: Running tasks the outer session answered: the controller ended their turn,
        #: and the answer, not the turn's end, is what becomes of the attempt.
        self._ending: set[str] = set()
        #: How long the loop stays for an answer once everything left waits for one, and
        #: until when it does (monotonic); ``None`` while something can still go on.
        self.answer_wait_s = ANSWER_WAIT_S if answer_wait_s is None else answer_wait_s
        self._leaves_at: float | None = None
        #: The final pass in flight on a frozen candidate, as a future of what it found
        #: (:class:`FinalPass`). While it is, nothing starts: nothing lands into the candidate.
        self._final: Future | None = None
        #: The last final pass that was judged; ``None`` before one.
        self.final: FinalPass | None = None
        #: That pass as the closing report says it (:meth:`FinalPass.as_record`), or why
        #: none stands for the target as it is; ``None`` while this process judged none.
        #: The report is its one record on disk: the ledger has each check's row.
        self.final_record: dict[str, Any] | None = None
        #: What the card's check of a task that just landed could not verify, by task id,
        #: until the loop records the attempt.
        self._landing_said: dict[str, str] = {}
        self._check_pool: ThreadPoolExecutor | None = None
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

    def _save_graph(self) -> None:
        self._persist(self.graph_path.name, self._write_graph)

    def _write_graph(self) -> None:
        """Write ``graph.yaml`` whole (:meth:`~.graph.Graph.save`): a monitor may be reading it."""
        self.graph.save(self.graph_path)

    # ----------------------------------------------------------------- checks
    def _check_by_key(self, key: str) -> contract_mod.Check | None:
        return next((c for c in self.contract.checks if c.key == key), None)

    def _check_lock(self, key: str) -> threading.RLock:
        with self._check_locks_guard:
            return self._check_locks.setdefault(key, threading.RLock())

    def _row(
        self, check: contract_mod.Check, key: str, since: int = 0
    ) -> contract_mod.Verdict | None:
        """The latest row of ``check`` recorded under ``key``; ``None`` when it has none.

        ``since`` counts only the rows recorded from that position of the ledger on.
        """
        with self._status_lock:
            return contract_mod.row(self.status.verdicts[since:], check, key)

    def _run_check(self, check: contract_mod.Check) -> bool:
        """One check, in a pool thread, recorded under its key on the target as it started.

        A check that has a row for that very key which did not fail (it passed, or it
        could not verify what it is for) does not run again: the same content says the
        same. That is asked as the check starts, and the key is read then. Nothing else
        is run for it: a check that needs another's result (:attr:`Check.after`) is a
        task that waits for the other's task.
        """
        with self._check_lock(check.key):
            return self._run_check_once(check)

    def _run_check_once(self, check: contract_mod.Check) -> bool:
        # What a worker's own run of this check left for this very key counts.
        self._reload_status()
        key = self.s.key(check, "")
        if key:
            held = self._row(check, key)
            if held is not None and held.passed:
                return True
        result = self.s.check(check)
        with self._status_lock:
            self.status.record(check.unit, result, marker=key, inputs=check.inputs)
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
        bounded pool. Each result is recorded under the key of what it measured: a
        check that overlapped another task's landing is about the tree it measured.
        """
        keys = list(keys)
        if self._check_pool is None:
            # Outside :meth:`run`, the checks get a pool of their own.
            with ThreadPoolExecutor(max_workers=self.max_parallel_checks) as pool:
                return _failed(self._submit_checks(pool, keys))
        return _failed(self._submit_checks(self._check_pool, keys))

    def _run_task(self, task: graph_mod.Task) -> tuple[str, list[str]]:
        """A task's worker, then, in the same thread, the contract checks it claims.

        Its agent turn is over before its checks start: the turn goes back to the counter.
        """
        try:
            decision = self.s.work(task)
        except Exception as exc:  # a turn that raised is a failed attempt, not a crashed run
            return f"raised {type(exc).__name__}: {exc}", []
        if decision.partition("\n")[0] not in FINISHED:
            return decision, []
        self._checking.add(task.id)
        try:
            return decision, self._run_checks(task.satisfies)
        except ChecksRaised as exc:
            # The work stands and has landed; only these checks' outcome is unknown.
            # They have no row: the final pass measures them.
            self._inconclusive[task.id] = str(exc)
            return decision, exc.failed
        finally:
            self._checking.discard(task.id)

    def demand(self) -> dict[int, float]:
        """The shapes the pool is asked for now, each with its longest wall: what is ready or running.

        A count of nodes to the seconds of the longest job of that shape. A row's task, the
        survey and the kernel fan-out ask one node, with the row's wall, and so do a
        deployment's assembly and its session while either is due. A deployment of whole
        nodes past one asks one node here too: its nodes are asked for by its session's
        job while that job is in the spool (``pool.spool_demand``), and are given back
        when no such job is left, so they are held while they work and not while a worker
        writes or repairs. A task that is blocked, skipped, done or still waiting for
        another asks nothing.
        Read by the pool's keeper from its own thread: the graph is only read.
        """
        tasks = list(self.graph.tasks)
        by_id = {t.id: t for t in tasks}
        due = [
            t
            for t in tasks
            if t.status == "running"
            or (t.status == "pending" and all(graph_mod.met(by_id.get(d)) for d in t.depends_on))
        ]
        if not due:
            return {}
        deployments = {d.name: d for d in self.s.deployments()}
        rows = {row.id: row for row in rows_mod.load_work(self.workspace)}
        owed: dict[int, float] = {}
        for task in due:
            name, _, what = task.check.partition(":")
            if name == rows_mod.DEPLOYMENT_CHECK and what in deployments:
                nodes = max(1, deployments[what].nodes)
                wall = float(deployments[what].wall or rows_mod.DEFAULT_WALL_S)
            elif task.kind == graph_mod.CHECK_KIND:
                continue  # a join runs nothing
            elif task.ref == rows_mod.DELIVERY_ROW:
                continue  # no job proves it: it asks the pool for no holder
            else:
                row = rows.get(task.ref)
                nodes = max(1, row.nodes) if row is not None else 1
                wall = float(row.wall if row is not None else rows_mod.DEFAULT_WALL_S)
            if nodes > 1:
                # Until its job is in the spool, work of whole nodes needs one for its own.
                nodes, wall = 1, float(rows_mod.DEFAULT_WALL_S)
            owed[nodes] = max(owed.get(nodes, 0.0), wall)
        return owed

    @contextlib.contextmanager
    def checking(self, task_id: str) -> Iterator[None]:
        """While open, running ``task_id`` holds no agent turn: the coordinator checks its work."""
        self._checking.add(task_id)
        try:
            yield
        finally:
            self._checking.discard(task_id)

    def void_passes(self, keys: Sequence[str], why: str) -> list[str]:
        """A requirement changed: the standing pass of each of ``keys`` no longer counts.

        The ledger gets a line that says why (``evidence.void_passes``), under the key
        the pass was recorded on, and the check is measured again when its task next runs.
        """
        self._reload_status()
        with self._status_lock:
            return evidence_mod.void_passes(self.status, keys, why)

    def _reload_status(self) -> None:
        """Take the ledger as it is on disk: a worker's own gate run appends to it too.

        The file only grows, so what it holds is what this process recorded and what
        other processes did; a read that fails leaves what this process has.
        """
        try:
            fresh = contract_mod.ContractStatus.load(self.workspace)
        except (OSError, ValueError, TypeError):
            return
        with self._status_lock:
            if len(fresh.verdicts) >= len(self.status.verdicts):
                self.status.verdicts[:] = fresh.verdicts

    def mark_integrated(self, task_id: str, attempt: int, not_verified: str = "") -> None:
        """``task_id``'s commits are on the target; ``not_verified`` is what its card's check could not verify.

        Kept until the loop records how the attempt ended (:meth:`_done`): a landing whose
        check failed nothing and could not observe what it is for says so on its task.
        Nothing is measured again for a landing: a record never starts work.
        """
        if task_id in self.graph.by_id():
            # Read by the loop's thread only; a landing the loop sees one pass late is still seen.
            self._landing_said[task_id] = not_verified

    def standing(self) -> contract_mod.Evaluation:
        """The contract on the target as it is: each check by its row there."""
        keys = {check.key: self.s.key(check, "") for check in self.contract.checks}
        with self._status_lock:
            return contract_mod.evaluate(self.contract, self.status, keys=keys)

    def completion(self, files: Sequence[str]) -> Completion:
        """Delivery reads the rows of the target as it is, the graph and ``files`` together."""
        return assess(
            self.standing(),
            files=list(files),
            # A required task that is blocked or skipped did not produce what the run owes:
            # a contract that holds all the same does not make the run delivered.
            required_open=self._required_open() or bool(self._required_unmet()),
        )

    def _required_open(self) -> bool:
        """Whether a required task is pending or running."""
        return any(
            t.priority == "required" and t.status in ("pending", "running")
            for t in self.graph.tasks
        )

    # ------------------------------------------------------------------- loop
    def _required_unmet(self) -> list[str]:
        """The required tasks that ended without their outcome: blocked, or skipped."""
        return [
            t.id
            for t in self.graph.tasks
            if t.priority == "required" and t.status in ("blocked", "skipped")
        ]

    def _finish(self, task: graph_mod.Task, decision: str, failed: list[str]) -> None:
        """Record how one attempt of ``task`` ended: done, to be tried again, or blocked."""
        decision, _, note = decision.partition("\n")
        note = note.strip()
        landing = self._landing_said.pop(task.id, "")
        if decision in FINISHED and not failed:
            self._done(task, landing)
        elif decision in FINISHED:
            self._checks_failed(task, failed, note)
        elif decision == BLOCKED:
            self._block(
                task, escalation_mod.WORKER, "blocked", note or "its worker's receipt says blocked"
            )
        elif decision.startswith(f"raised {TaskNotStarted.__name__}:"):
            # No worker ran, so no attempt is spent and no agent can do better: the tree
            # could not be prepared (a full filesystem, git), which is the environment's.
            self._block(task, escalation_mod.INTEGRATION, "tree_not_prepared", decision)
        else:
            said = decision or "its turn ended with no receipt"
            self._attempt_failed(task, f"{said}: {note}" if note else said)
        self._save_graph()
        self._persist("state", self.s.checkpoint)
        self.s.finished(task)

    def _not_verified(self, task: graph_mod.Task) -> list[str]:
        """The contract checks ``task`` claims that failed nothing and could not be verified, each with why.

        As the ledger has them on the target as it is now.
        """
        evaluation = self.standing()
        return [
            f"{check.key}: {verdict.unverified}"
            for check, verdict in evaluation.unverified
            if check.key in task.satisfies
        ]

    def _done(self, task: graph_mod.Task, landing: str = "") -> None:
        """``task`` is done; ``landing`` is what its card's check could not verify as it landed."""
        task.status, task.last = "done", ""
        # A doubt its work did not wait for (a requirement that looks wrong): the task is
        # done, and its question goes to the outer session and holds nothing. A whole run's
        # survey blocked on such a doubt and ran again from the start after the answer.
        try:
            asked = str((self.s.receipt(task) or {}).get("next_question") or "").strip()
        except Exception:  # an unreadable receipt asks nothing
            asked = ""
        task.block = (
            {"by": escalation_mod.WORKER, "kind": "question", "question": _clip(asked, 600)}
            if asked
            else {}
        )
        if asked:
            self.s.log(f"[asks] {task.id} ({task.kind}), done, holds nothing: {_clip(asked, 300)}")
        unknown = self._inconclusive.pop(task.id, "")
        # Done, and it says so: nothing in the target repairs a check that could not observe
        # what it is for, so no attempt is spent on it and the run closes ``partial``.
        task.unverified = ([landing] if landing else []) + self._not_verified(task)
        self.s.log(
            f"[task] {task.id} ({task.kind}) done on attempt {task.attempts}"
            + (
                f"; some checks could not run ({unknown}): the final pass measures them"
                if unknown
                else ""
            )
            + (f"; NOT VERIFIED: {'; '.join(task.unverified)}" if task.unverified else "")
        )
        if task.kind == "survey":
            self._expand()
        self._publish_onboarded(task)

    def _publish_onboarded(self, task: graph_mod.Task | None) -> None:
        """Pin ``onboarded`` when ``task`` is the first deployment's session and it is done.

        The publication is the service's and happens once (it returns what stands when
        asked again). One that failed is owed: it is asked again at the next start.
        """
        if task is None or self.s.onboarded is None:
            return
        if task.status != "done" or task.id != self._first_session() or task.onboarded:
            return
        try:
            record = self.s.onboarded(task)
        except Exception as exc:  # the record is owed again at the next start
            self.s.log(f"[onboarded] the ref could not be pinned: {type(exc).__name__}: {exc}")
            return
        # The link to what was published: with it, a later start owes nothing.
        commit = record.get("commit") if isinstance(record, Mapping) else None
        task.onboarded = str(commit or "published")

    def _attempt_failed(self, task: graph_mod.Task, why: str) -> None:
        """One attempt of ``task`` did not end done: the next one starts, or the task is blocked."""
        why = _clip(why)
        if task.attempts < task.budget:
            self.s.log(
                f"[retry] {task.id} ({task.kind}) attempt {task.attempts} of {task.budget} "
                f"ended: {why}; it starts again"
            )
            task.attempts += 1
            task.status, task.last = "pending", why
            return
        self._block(task, escalation_mod.GATE, "attempts_exhausted", why)

    def _checks_failed(self, task: graph_mod.Task, failed: Sequence[str], note: str = "") -> None:
        """``task``'s work landed and its checks did not pass.

        A worker's task tries again with what failed. A ``check`` task has no worker: what
        it measured goes back to the workers' tasks it follows directly, each for one more
        attempt, and it runs again after them. With nobody to hand it to, or no attempt
        left, it is blocked for the outer session.
        """
        unknown = self._inconclusive.pop(task.id, "")
        why = (f"{note}: " if note else "its checks failed: ") + ", ".join(failed)
        if unknown:
            why += f"; some checks could not run ({unknown})"
        if task.kind != graph_mod.CHECK_KIND:
            self._attempt_failed(task, why)
            return
        tasks = self.graph.by_id()
        owners = [
            tasks[dep]
            for dep in task.depends_on
            if dep in tasks
            and tasks[dep].status == "done"
            and tasks[dep].kind not in (graph_mod.CHECK_KIND, "survey")
        ]
        spent = [o.id for o in owners if o.attempts >= o.budget]
        if not owners or spent or task.attempts >= task.budget:
            self._block(task, escalation_mod.GATE, "check_failed", why)
            return
        for owner in owners:
            owner.attempts += 1
            owner.status = "pending"
            owner.last = _clip(
                f"`{task.id}` measured what your attempt {owner.attempts - 1} landed, and "
                f"{why}. Its lines are in the run's gate records; find the cause before you "
                f"change anything"
            )
        task.attempts += 1
        task.status, task.last = "pending", _clip(why)
        self.s.log(
            f"[retry] {task.id} (check) {why}; back to {', '.join(o.id for o in owners)}, "
            f"and it runs again after them"
        )

    def _block(
        self, task: graph_mod.Task, source: str, kind: str, observed: str, ask: str = ""
    ) -> None:
        """``task`` cannot go on: it waits for the outer session; what does not depend on it goes on.

        Its question is its worker's (the receipt's ``next_question``), else ``ask``, else
        what the loop asks of every block of that kind (:data:`ASKS`): a block never
        leaves its reader to work out which verbs answer it. The graph keeps who blocked
        it, as what, and the question; the receipt, written once, stays the worker's.
        """
        try:
            receipt = dict(self.s.receipt(task) or {})
        except Exception:  # an unreadable receipt is no reason to lose the block
            receipt = {}
        question = (
            receipt.get("next_question")
            or ask
            or ASKS.get(kind, "").format(task=task.id, budget=task.budget)
        )
        blocks = graph_mod.blocked_by(self.graph, task.id)
        task.status, task.last = "blocked", _clip(observed)
        task.block = {
            "by": source,
            "kind": str(receipt.get("blocker_kind") or kind),
            "question": _clip(str(question), 600),
        }
        self.s.log(
            f"[blocked] {task.id} ({task.kind}) on attempt {task.attempts}, "
            f"{task.block['kind']}: {_clip(observed, 300)}"
            + (f"; it holds {', '.join(blocks)}" if blocks else "")
            + "; everything else goes on"
        )

    # ------------------------------------------------------- tasks, by code
    def _first_session(self) -> str:
        """The id of the first deployment's session task; empty with a baseline or no deployment."""
        deployments = list(self.s.deployments())
        if self.baseline or not deployments:
            return ""
        return session_task_id(deployments[0])

    def holds_onboarded(self, task_id: str) -> bool:
        """Whether ``onboarded`` waits for ``task_id``'s landing: the first deployment's assembly.

        Until it is pinned. That landing is the last thing before the first session reads
        the assembly's passing result, so a turn spent there on anything but correctness
        is a turn the milestone waits.
        """
        tasks = self.graph.by_id()
        session, task = tasks.get(self._first_session()), tasks.get(task_id)
        if session is None or task is None or session.status == "done" or session.onboarded:
            return False
        return task.kind == "assemble" and task_id in session.depends_on

    def _expand(self) -> None:
        """Add the tasks the survey's rows and the spec's deployments make, with no agent turn.

        Idempotent: a task the graph already holds is left as it is, so a later start adds
        only what a changed spec added. The rows the tasks work on are kept in
        ``work.yaml``; before any of them starts, the package skeleton on the way to their
        files is committed (:attr:`Services.skeleton`), so no two tasks create one
        initializer.
        """
        path = self.workspace / vocab_mod.VOCABULARY_FILENAME
        rows: list[rows_mod.Row] = []
        raw_queue: dict[str, vocab_mod.QueueItem] = {}
        if path.is_file():
            try:
                rows = rows_mod.load_rows(path)
                raw_queue = {str(item.n): item for item in vocab_mod.load_vocabulary(path)}
            except vocab_mod.VocabularyError as exc:
                self.s.log(f"[tasks] no task was made from the survey: {exc}")
                return
        own = self.validation["own_paths"]
        survey = next((t for t in self.graph.tasks if t.kind == "survey"), None)
        compiled = compile_tasks(
            rows,
            list(self.s.deployments()),
            queue=raw_queue,
            target_rel=(survey.ref if survey is not None else "") or (own[0] if own else ""),
            own_paths=own,
            allowed_paths=self.validation["allowed_paths"],
            repo_root=self.repo_root,
            baseline=bool(self.baseline),
            budget=self.max_attempts,
            required_checks=self.validation["required_checks"],
            survey=survey.id if survey is not None else "",
            ci=tuple(self.s.ci_files()),
        )
        known = self.graph.by_id()
        added = [t for t in compiled.tasks if t.id not in known]
        rows_mod.save_work(self.workspace, compiled.rows)
        self.graph.tasks.extend(added)
        self._wait_for_rows(compiled.tasks, known)
        if not added:
            return
        for task in added:
            if task.id in compiled.unplaced:
                self._block(task, escalation_mod.GATE, "write_scope", compiled.unplaced[task.id])
        # A row the run may not commit yet gets its directories when it is let in.
        self._skeleton([t for t in added if t.id not in compiled.unplaced])
        problems = graph_mod.validate(
            self.graph, **{k: v for k, v in self.validation.items() if k != "own_paths"}
        )
        for problem in problems:
            self.s.log(f"[tasks] {problem}")
        self.s.log(
            f"[tasks] {len(rows)} row(s) and {len(list(self.s.deployments()))} deployment(s) "
            f"made {len(added)} task(s): {', '.join(t.id for t in added)}"
        )

    def _wait_for_rows(
        self, compiled: Sequence[graph_mod.Task], known: Mapping[str, graph_mod.Task]
    ) -> None:
        """Give each task that is not done the dependencies the rows give it now.

        A later survey may add a row that a task the graph already holds must wait for
        (a target's row, for the first assembly): left as it was, that task started
        without the row's work. Dependencies are only added, never taken away: the same
        list holds what an answer put there (``retry --task``, a split). One that would
        close a circle is left out and said. A task that runs is not stopped: it is said,
        and its next attempt waits. The rest of a task (its write scope, its check, its
        goal) stays as it was made: answers and a turn's own rules add to those too,
        and a row that changes them under a task is not something a run has shown.
        """
        for fresh in compiled:
            task = known.get(fresh.id)
            if task is None or task.status in ("done", "skipped"):
                continue
            more = [d for d in fresh.depends_on if d not in task.depends_on]
            circle = [d for d in more if d in graph_mod.blocked_by(self.graph, task.id)]
            more = [d for d in more if d not in circle]
            task.depends_on += more
            if more:
                self.s.log(
                    f"[tasks] {task.id} now waits for {', '.join(more)} too: the survey's rows "
                    "say so"
                    + (
                        ". It is running and is not stopped: its next attempt waits (`resolve "
                        f"retry {task.id} --note ...` ends this one)"
                        if task.status == "running"
                        else ""
                    )
                )
            if circle:
                self.s.log(
                    f"[tasks] {task.id} is not made to wait for {', '.join(circle)}, which "
                    "the survey's rows now ask: that task already waits for this one"
                )

    def recontract(self, contract: contract_mod.Contract) -> None:
        """The spec changed under a running controller (``resolve``): it executes the new contract.

        The checks it evaluates, runs and requires are the changed spec's from this pass on;
        a result recorded with a check's old inputs is not that check's result any more.
        """
        self.contract = contract
        self.validation["required_checks"] = [c.key for c in contract.required()]
        self.validation["known_checks"] = [c.key for c in contract.checks]

    def allow_paths(self, paths: Iterable[str]) -> None:
        """The paths the run may commit changed (``resolve allow-shared``): tasks made from now on are held to these."""
        self.validation["allowed_paths"] = tuple(paths)

    def _skeleton(self, tasks: Sequence[graph_mod.Task]) -> set[str]:
        """Commit the package skeleton on the way to the files ``tasks`` write; who cannot start.

        Asked as the task list is made, for every row that can start, and again as each
        task starts (a row a resolution let in, a task a split made, a run started again).
        Nothing is remembered here: what is owed is read off the repository each time, and
        asking for what exists commits nothing. A skeleton that could not be committed
        stops the tasks that need it, with the reason: their cards promise the shared
        initializers are in place, and two of them creating one would conflict at landing.
        """
        owed = [t for t in tasks if any("." in p.rsplit("/", 1)[-1] for p in t.touches)]
        files = sorted({p for t in owed for p in t.touches if "." in p.rsplit("/", 1)[-1]})
        if self.s.skeleton is None or not files:
            return set()
        try:
            said = self.s.skeleton(files)
        except Exception as exc:
            why = (
                f"the package skeleton its files need was not committed "
                f"({type(exc).__name__}: {exc})"
            )
            self.s.log(f"[skeleton] {why}: {', '.join(t.id for t in owed)} do not start")
            for task in owed:
                if task.status in ("pending", "running"):
                    self._block(task, escalation_mod.INTEGRATION, "skeleton_failed", why)
            return {t.id for t in owed}
        if said and not said.startswith(NOTHING_OWED):
            self.s.log(f"[skeleton] {said}")
        return set()

    def _split_tasks(
        self, raw: Any, parent: graph_mod.Task
    ) -> tuple[list[graph_mod.Task], Callable[[], None]]:
        """The tasks of the typed rows a split names for ``parent`` (``resolve split``)."""
        return split_tasks(
            raw,
            parent,
            self.graph,
            workspace=self.workspace,
            own_paths=self.validation["own_paths"],
            allowed_paths=self.validation["allowed_paths"],
            repo_root=self.repo_root,
            budget=self.max_attempts,
        )

    # ------------------------------------------------- resolutions, the pool
    def _resolutions(self) -> None:
        """Apply the answers the outer session left since the last pass, each once."""
        base = self.s.hooks() if self.s.hooks is not None else escalation_mod.Hooks()
        base.make_tasks = self._split_tasks
        base.void = self.void_passes
        try:
            escalation_mod.apply_pending(
                self.workspace, self.graph, base, self._write_graph, self.s.log, self._hold
            )
        except OSError as exc:
            if exc.errno not in STORAGE_FULL:
                raise
            # The change is in memory: it lands with the next save, and the answer is then
            # found stale and settled.
            self._save_graph()

    def _hold(self, answer: Mapping[str, Any]) -> bool:
        """Whether ``answer`` waits a pass: it names a task whose thread still runs.

        `retry` and `skip` on a running task end its agent turn first (the same stop as
        the turn's own limit), and are applied once its thread has returned. A task
        whose turn is over (its checks run) has no turn to end: the answer waits for
        its checks and is then judged against how the attempt ended.
        """
        name = str(answer.get("name") or "")
        task = self.graph.by_id().get(name)
        if (
            name not in self._running
            or task is None
            or task.status != "running"
            or answer.get("verb") not in escalation_mod.ENDS_A_TURN
        ):
            return False
        if self.s.end_turn(task) and name not in self._ending:
            self.s.log(
                f"[answer] {name}: the outer session answered with `{answer.get('verb')}` "
                f"while its turn ran (attempt {task.attempts}): the turn is ended"
            )
        self._ending.add(name)
        return True

    def _start_check_task(self, task: graph_mod.Task) -> Future:
        """Submit a ``check`` task's checks to the check pool; the future of its outcome.

        No agent turn runs and none is counted: its checks wait for
        devices in the spool as every check does. One with no check of its own is a
        join, done as it starts.
        The future holds what a worker's thread returns, :data:`CHECKED` and
        the keys that failed, once every check is in.
        """
        submitted = self._submit_checks(self._check_pool, task.satisfies)
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

    def run(self) -> str:
        """Drive the run to a verdict: ``delivered``, ``escalated`` or ``partial``.

        Or ``stopped: <reason>``, when :func:`stoppable` asked it to stop: it
        then returns at once, and the turns and checks in flight are left to
        end on their own. The loop's own failure, or Ctrl-C, records what
        finished and raises at once, the same way; a resumed run starts them again.
        """
        checks = ThreadPoolExecutor(
            max_workers=self.max_parallel_checks, thread_name_prefix="mv2-check"
        )
        # A task's thread outlives its agent turn while its checks run: room for those.
        tasks = ThreadPoolExecutor(
            max_workers=2 * self.max_parallel + 2, thread_name_prefix="mv2-task"
        )
        self._check_pool = checks
        crashed = False
        try:
            try:
                return self._drive(tasks)
            except (Exception, KeyboardInterrupt):
                # The loop's own failure, or Ctrl-C: what finished is recorded before the process
                # goes, and nothing still running is waited for; a resumed run starts it again.
                crashed = True
                self._record_finished()
                raise
            finally:
                stopped = crashed or self._stop.done()
                tasks.shutdown(wait=not stopped, cancel_futures=stopped)
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

    def _drive(self, tasks: ThreadPoolExecutor) -> str:
        running: dict[str, Future] = {}
        self._running = running
        continued = False
        # A survey that finished before this start still owes its tasks, a spec that gained
        # a deployment owes that deployment's, and a run that starts with no survey (its
        # vocabulary was written before, or it has none) has only what code makes.
        if not any(
            t.kind == "survey" and t.status in ("pending", "running") for t in self.graph.tasks
        ):
            self._expand()
            self._save_graph()
        # A first session that passed before this start has its publication, or is owed it.
        first = self.graph.by_id().get(self._first_session())
        if first is not None and first.status == "done" and not first.onboarded:
            self._publish_onboarded(first)
            self._save_graph()
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
                    except Exception as exc:  # the run's own failure, not the task's
                        decision, failed = f"raised {type(exc).__name__}: {exc}", []
                    if tid in self._ending and decision.startswith(f"raised {TurnEnded.__name__}:"):
                        # Ended for an answer: the answer is what becomes of the attempt.
                        continue
                    self._ending.discard(tid)
                    self._finish(task, decision, failed)
            if not self._written():
                # A full filesystem: what runs goes on and is recorded here; nothing starts.
                self._wait(running, self.poll_s)
                continue
            self._resolutions()
            for tid in sorted(self._ending - set(running)):
                # Its turn was ended and no answer took the attempt (a later answer
                # replaced the one it was ended for): it runs again as the attempt it was.
                self._ending.discard(tid)
                task = self.graph.by_id().get(tid)
                if task is not None and task.status == "running":
                    task.status = "pending"
                    self._resumed.add(tid)
                    self._save_graph()
            # One counter of agent turns: a check task takes none, nor a task whose turn
            # is over and whose checks run. Devices are the pool's to hand out.
            kinds = {t.id: t.kind for t in self.graph.tasks}
            turns = sum(
                1
                for tid in running
                if kinds.get(tid) != graph_mod.CHECK_KIND and tid not in self._checking
            )
            slots = max(self.max_parallel - turns, 0)
            self._resumed &= {t.id for t in self.graph.tasks if t.status == "pending"}
            # While the final pass runs the candidate is frozen: nothing starts, so nothing
            # lands into it. What a resolution reopens meanwhile starts once the pass is in.
            chosen = (
                graph_mod.next_to_start(self.graph, resumed=self._resumed)
                if self._final is None
                else []
            )
            started = [t for t in chosen if t.kind == graph_mod.CHECK_KIND] + [
                t for t in chosen if t.kind != graph_mod.CHECK_KIND
            ][:slots]
            unprepared = self._skeleton([t for t in started if t.kind != graph_mod.CHECK_KIND])
            started = [t for t in started if t.id not in unprepared]
            for task in started:
                task.status = "running"
                self._resumed.discard(task.id)
            if started or unprepared:
                # Before any work starts: a task is recorded as running, then it runs. A
                # process that dies in between finds a running task with nothing done.
                self._save_graph()
            if started and not self._written():
                # A full filesystem: nothing starts unrecorded; the next pass tries again.
                for task in started:
                    task.status = "pending"
                started = []
            for task in started:
                if task.kind == graph_mod.CHECK_KIND:
                    running[task.id] = self._start_check_task(task)
                else:
                    running[task.id] = tasks.submit(self._run_task, task)
            # What runs waits on jobs that cannot start before this host's job ends, and no
            # agent turn is in flight that could change it: "to be continued" is said now,
            # not at the job's limit. (With nothing running the pool says it itself.)
            late = "" if continued or turns or not running else self.s.late()
            if late and all(t.kind == graph_mod.CHECK_KIND for t in started):
                continued = True  # asked once: where no job's end is watched, nothing follows
                if controller_mod.continue_now(self.workspace, late):
                    self.s.log(f"[stop] to be continued: {late}")
            verdict = ""
            if self._final is not None and self._final.done():
                verdict = self._final_done()
            elif not running and self._final is None:
                verdict = self._spent(tasks)
            if verdict != "escalated":
                self._leaves_at = None
            elif self._stays_for_answer():
                verdict = ""  # an answer still reaches this controller: look again
            if verdict:
                return verdict
            self._wait(running, self.poll_s)

    def _stays_for_answer(self) -> bool:
        """Whether this controller still stays for an answer; said once, when the stay begins.

        Everything left waits for the outer session. Its answer comes minutes later (it
        looks, reads a receipt, decides), and one that finds no controller is applied by
        ``resolve`` itself and then needs the run started again: a wait in a queue, with
        the run's holders idle beside it. So the controller stays :data:`ANSWER_WAIT_S`
        and takes the answer in the job it has. ``escalated`` and its exit code mean what
        they meant, that much later.
        """
        now = time.monotonic()
        if self._leaves_at is None:
            self._leaves_at = now + self.answer_wait_s
            if self.answer_wait_s > 0:
                asked = [t.id for t in self.graph.tasks if t.status == "blocked"]
                until = time.strftime("%H:%M:%SZ", time.gmtime(time.time() + self.answer_wait_s))
                self.s.log(
                    "[loop] everything left waits for an answer "
                    f"({', '.join(asked) or escalation_mod.RUN}): this controller is alive and "
                    f"takes one until {until} ({self.answer_wait_s / 60:.0f} min), then leaves "
                    "with exit 76. An answer after that needs the run started again"
                )
        return now < self._leaves_at

    def _spent(self, tasks: ThreadPoolExecutor) -> str:
        """Nothing runs and nothing could start: ``escalated``, or the final pass starts.

        With a task blocked, ``escalated``: everything left waits for the outer session,
        and no final pass runs on a target a resolution is about to change. So it is while
        an escalation of the run itself stands open (a final pass that did not hold). With
        every task done or skipped, the target is frozen and the one final pass starts, in
        a thread of its own: the loop goes on serving stops, resolutions and the pool.
        ``""`` to look again.
        """
        if graph_mod.ready(self.graph):
            return ""  # something can still start: look again
        blocked = [t.id for t in self.graph.tasks if t.status == "blocked"]
        if blocked:
            if self._leaves_at is None and self.final_record is not None:
                # No pass stands for the target as it is: an earlier one's record goes.
                # Not while the controller stays for an answer to what that pass found.
                self.final_record = {
                    "candidate": "",
                    "said": f"not run: {', '.join(blocked)} blocked; the final pass runs "
                    "once every task is done or skipped",
                }
            return "escalated"
        if self.graph.run_blocked:
            return "escalated"
        self.s.log(
            "[final] every task is done or skipped: the target is frozen, and the final pass "
            "starts on that candidate"
        )
        self._final = tasks.submit(self._final_pass)
        return ""

    def _final_pass(self) -> FinalPass:
        """The one pass on the frozen candidate: every requested check has a row whose key is its.

        The candidate is the target's head, pinned under the landing lock
        (:attr:`Services.freeze`); the run lands nothing into it while this runs, and
        what anything else moved is seen when it ends (:meth:`_moved_since`). Each requested
        check is asked for its key on the candidate. One that has a row under that very
        key is not measured again, whatever commit the row was measured on; one that has
        none (its key changed since it was measured, or it never ran) is measured now,
        once. A check whose task is not done (skipped, or held behind a skipped one) is
        not run and is reported so. A check judged against another's result is measured
        after it, however long the chain (:func:`_waves`). Runs in a thread of its own; it
        changes nothing but the ledger.
        """
        contract = self.contract
        candidate = self.s.freeze() if self.s.freeze is not None else ""
        final = FinalPass(candidate, contract)
        owners = {key: (t.id, t.status) for t in list(self.graph.tasks) for key in t.satisfies}
        self._reload_status()
        required = contract.required()
        reused: list[str] = []
        measured: list[str] = []
        for wave in _waves(required):
            owed = []
            for check in wave:
                task_id, status = owners.get(check.key, ("", "done"))
                if status != "done":
                    final.waived[check.key] = f"its task `{task_id}` is {status}"
                    continue
                key = self.s.key(check, candidate)
                held = self._row(check, key) if key else None
                final.rows[check.key] = FinalRow(key, held, False)
                if held is None:
                    owed.append(check)
                else:
                    reused.append(check.key)
            if not owed:
                continue
            with self._status_lock:
                since = len(self.status.verdicts)
            try:
                self._run_checks([c.key for c in owed])
            except ChecksRaised as exc:
                self.s.log(f"[final] checks raised and left no result: {exc}")
            for check in owed:
                key = final.rows[check.key].key
                final.rows[check.key] = FinalRow(key, self._row(check, key, since), True)
                measured.append(check.key)
        self.s.log(
            f"[final] candidate {candidate[:10] or '(unnamed)'}: {len(required)} requested "
            f"check(s); not measured again, a row has the candidate's key: "
            f"{', '.join(reused) or 'none'}; measured now: {', '.join(measured) or 'none'}"
            + (
                "; not run: " + "; ".join(f"{k} ({why})" for k, why in final.waived.items())
                if final.waived
                else ""
            )
        )
        evaluation = final.evaluation()
        if self.s.final_tests is None:
            final.tests_said = "this run has no final tests"
        elif evaluation.failed or evaluation.missing:
            final.tests_said = (
                "not run: a requested check did not pass on this candidate, and its repair "
                "makes a new one"
            )
        else:
            # One job of tests, once, on the candidate the checks hold on.
            try:
                final.tests = self.s.final_tests(candidate)
            except Exception as exc:  # said, never taken for a pass
                final.tests_lost = True
                final.tests_said = f"not run: {type(exc).__name__}: {exc}"
            else:
                if final.tests is None:
                    final.tests_said = "nothing to run: no test file is selected"
                else:
                    self.s.log(f"[final] final tests: {final.tests.summary()}")
        final.moved = self._moved_since(final)
        return final

    def _moved_since(self, final: FinalPass) -> str:
        """What of the target is no longer as the pass froze it; ``""`` when it all stands.

        The run itself lands nothing while the pass runs (the loop starts no task then).
        The session that watches it can: a commit made by hand in the checkout, a library
        built again in place. One comparison when the pass ends, with no lock held in
        between: the head, and each requested check's key there, against what the pass
        read its rows under.
        """
        head = self.s.freeze() if self.s.freeze is not None else final.candidate
        if head != final.candidate:
            return f"the target's head is {head[:10] or '(unnamed)'} now"
        changed = [
            check.key
            for check in final.contract.required()
            if (found := final.rows.get(check.key)) is not None
            and found.key
            and self.s.key(check, head) != found.key
        ]
        if changed:
            return f"the key of {', '.join(changed)} is another now (a build or an input changed)"
        return ""

    def _final_done(self) -> str:
        """The final pass is in: the run's verdict, or ``""`` when the run moved on meanwhile.

        ``delivered`` when every requested check passed on the candidate and its delivery
        files check out. ``partial`` when nothing failed and a task was skipped, a
        requested check was not run for it, or a check could not be verified: nothing in
        the target repairs those, so nothing is escalated and the report lists them.
        ``escalated`` when a requested check failed there or left no result: its task is
        blocked on it, and a repair makes a new candidate. A resolution applied while the
        pass ran (a task reopened, a requirement changed) makes that candidate stale, and
        so does a target that is no longer as the pass froze it (:meth:`_moved_since`): it
        is not judged, and the next pass freezes a new one; rows whose keys still match
        are reused there.
        """
        future, self._final = self._final, None
        try:
            final: FinalPass = future.result()
        except Exception as exc:  # the pass itself broke: said, never taken for a pass
            self._block_run(
                "final_pass_failed",
                f"the final pass raised {type(exc).__name__}: {exc}",
                "the final pass runs to its end on the frozen candidate",
            )
            return "escalated"
        moved = (
            final.contract is not self.contract
            or bool(graph_mod.ready(self.graph))
            or any(t.status in ("blocked", "running") for t in self.graph.tasks)
        )
        if moved or final.moved:
            self.s.log(
                f"[final] the run changed while the final pass ran on {final.candidate[:10]} ("
                + (final.moved or "a resolution reopened work or changed a requirement")
                + "): that candidate is not judged, and the next pass freezes a new one"
            )
            return ""
        self.final = final
        evaluation = final.evaluation()
        where = f"the final pass on the candidate {final.candidate[:10] or '(unnamed)'}"
        for check, verdict in evaluation.failed:
            owner = next((t for t in self.graph.tasks if check.key in t.satisfies), None)
            observed = f"{where}: `{check.key}` failed: {verdict.detail}"
            owned = owner is not None and owner.status == "done"
            name = owner.id if owned else escalation_mod.RUN
            ask = (
                f"What repairs `{check.key}`? `resolve retry {name} --task <the task whose "
                "work broke it> --note <what to fix>` sends that task back, and its repair "
                f"is a new candidate; `resolve change-requirement {name}` if the failure is "
                f"the requirement's; `resolve retry {name}` alone measures the same "
                "candidate again."
            )
            if owned:
                self._block(owner, escalation_mod.GATE, "final_check_failed", observed, ask)
            else:
                self._block_run(
                    "final_check_failed", observed, f"`{check.key}` passes on the candidate", ask
                )
        lost = [c.key for c in evaluation.missing]
        if lost:
            self._block_run(
                "final_check_not_run",
                f"{where}: {', '.join(lost)} left no result there",
                "every requested check has a row whose key is the candidate's",
                "Why did its job leave no result (the journal's `gate` lines and the spool's "
                "log say)? Once that is put right, `resolve retry run --task <the session's "
                "task>` measures it again.",
            )
        if evaluation.failed or lost:
            self._final_closed(final, "escalated")
            return "escalated"
        # Tests that failed nothing and do not certify (every one skipped, a declared file
        # missing, a comparison with the base that is unknown) are the third outcome:
        # listed, never passed, and nothing is escalated for them.
        untested = final.tests is not None and final.tests.partial
        if final.tests_lost or (
            final.tests is not None and not final.tests.passed and not untested
        ):
            failures = final.tests.failures if final.tests is not None else []
            found = (final.tests.summary() if final.tests is not None else final.tests_said) + (
                "; " + "; ".join(failures) if failures else ""
            )
            if final.tests_lost or final.tests.verdict == "INCOMPLETE":
                # No case failed: a job ended without a result (its driver's own
                # traceback, a job that never started). The environment is looked at
                # first, and the question sends no task back.
                last = final.tests_said if final.tests is None else final.tests.job_said
                self._block_run(
                    "final_tests_not_run",
                    f"{where}: the final tests did not run to a result: {found}",
                    "every selected test has a result on the candidate",
                    "The final tests did not run to a result: no case failed, so no task's "
                    "work is shown at fault and none is sent back. "
                    + (f"The test job's last line: `{_clip(last, 300)}`. " if last else "")
                    + "Look at the environment first (the job's log in the spool has the "
                    "rest). Once that is put right, `resolve retry run --note <why>` runs "
                    "the final pass again on the target as it is.",
                )
            else:
                self._block_run(
                    "final_tests_failed",
                    f"{where}: the final tests did not pass: {found}",
                    "every selected test passes on the candidate, or fails at the base too",
                    "Which task's work broke each failing case? `resolve retry run --task "
                    "<that task> --note <the case and what to fix>` sends it back; its "
                    "repair is a new candidate, and the final tests run on that one. If no "
                    "task's work is at fault (a case that fails by chance), `resolve retry "
                    "run --note <why>` runs the final pass again on the target as it is.",
                )
            self._final_closed(final, "escalated")
            return "escalated"
        self._refresh_unverified(evaluation)
        if (
            evaluation.holds
            and not final.waived
            and not untested
            and self.s.delivery_files is not None
        ):
            missing = list(self.s.delivery_files())
            if missing:
                self._block_run(
                    "not_delivered",
                    "every requested check passed on the candidate, but the delivery files "
                    "on disk do not check out: " + "; ".join(missing),
                    "every requested check holds on the target, and its delivery files check out",
                    "Which task owns each file named? `resolve retry run --task <that task> "
                    "--note <the file and what is wrong with it>` sends it back.",
                )
                self._final_closed(final, "escalated")
                return "escalated"
        verdict = self._final_verdict(final)
        self._final_closed(final, verdict)
        return verdict

    def _final_verdict(self, final: FinalPass) -> str:
        """``partial`` or ``delivered``, once nothing of the final pass failed."""
        left = [t.id for t in self.graph.tasks if t.status == "pending"]
        # A skipped task that was only nice to have keeps nothing from delivered.
        skipped = [
            t.id for t in self.graph.tasks if t.status == "skipped" and t.priority == "required"
        ]
        if skipped or left or final.waived:
            self.s.log(
                f"[partial] skipped: {', '.join(skipped) or 'none'}"
                + (
                    f"; not started, behind a skipped required task: {', '.join(left)}"
                    if left
                    else ""
                )
                + (f"; requested checks not run: {', '.join(final.waived)}" if final.waived else "")
            )
            return "partial"
        landed = [
            f"{t.id}: {u}" for t in self.graph.tasks if t.status == "done" for u in t.unverified
        ]
        if final.tests is not None:
            landed += [f"final tests: {u}" for u in final.tests.unverified]
        if landed:
            self.s.log(
                "[partial] every task is done and nothing failed; not verified: "
                + "; ".join(landed)
            )
            return "partial"
        return "delivered"

    def _final_closed(self, final: FinalPass, verdict: str) -> None:
        """What the final pass found is kept for the report, and what it changed of the graph is written."""
        self.final_record = final.as_record(verdict)
        self._save_graph()
        self._persist("state", self.s.checkpoint)

    def _refresh_unverified(self, evaluation: contract_mod.Evaluation) -> None:
        """Each done task's "not verified" list, as the candidate has it.

        What a task listed of a requested check is as that check stood when the task
        ended. The final pass may have measured it again on a changed key: a check that
        passed on the candidate is taken off the list (its session's card check with it),
        and one that could not be verified there is on its task's list.
        """
        passed = {c.key for c in evaluation.passed}
        sessions = {
            f"{evidence_mod.SESSION}:{c.unit}"
            for c in evaluation.passed
            if c.gate == evidence_mod.SESSION
        }
        for task in self.graph.tasks:
            if task.status != "done":
                continue
            kept = [u for u in task.unverified if u.partition(": ")[0] not in passed | sessions]
            for check, verdict in evaluation.unverified:
                said = f"{check.key}: {verdict.unverified}"
                if check.key in task.satisfies and said not in kept:
                    kept = [u for u in kept if u.partition(": ")[0] != check.key] + [said]
            task.unverified = kept

    def _block_run(self, kind: str, observed: str, expected: str, ask: str = "") -> None:
        """The run itself asks the outer session: a final pass that did not hold, on no task.

        One question of the run at a time, kept in the graph until an answer empties it:
        what else the same pass found is added to what was observed. ``ask`` is its
        question for whoever answers it: which verb, on which task.
        """
        held = self.graph.run_blocked
        if held:
            held["observed"] = _clip(f"{held.get('observed')}; {observed}", 1200)
        else:
            self.graph.run_blocked = {
                "kind": kind,
                "observed": _clip(observed, 1200),
                "expected": expected,
                "question": ask,
            }
        self.s.log(f"[blocked] the run, {kind}: {_clip(observed, 300)}")
        self._save_graph()

    def _wait(self, running: dict[str, Future], timeout: float | None) -> None:
        """Until something in flight ends, the run is stopped, or ``timeout`` passes.

        A bounded wait lets pending interpreter signals run during long turns, and is
        how often the loop looks for a resolution command or a pool event.
        """
        in_flight = self._in_flight(running)
        timeout = self.poll_s if timeout is None else min(timeout, self.poll_s)
        wait([*in_flight, self._stop], timeout=timeout, return_when=FIRST_COMPLETED)

    def _in_flight(self, running: dict[str, Future]) -> list[Future]:
        """The tasks in flight, and the final pass when it is, as futures to wait on."""
        in_flight = list(running.values())
        if self._final is not None and not self._final.done():
            in_flight.append(self._final)
        return in_flight


def _clip(text: Any, limit: int = LAST_CHARS) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def seed(workspace: Path | str, tasks: Sequence[graph_mod.Task]) -> bool:
    """Start a run that has no graph yet with ``tasks`` (the survey); whether it was seeded.

    A workspace that has a graph is left as it is.
    """
    path = Path(workspace) / graph_mod.GRAPH_FILENAME
    if path.exists():
        return False
    graph_mod.Graph(tasks=list(tasks)).save(path)
    return True


def session_task_id(deployment: Deployment) -> str:
    """The task that runs a deployment's session: ``session-<name>``, or ``pd-hop-<name>``."""
    prefix = "pd-hop" if deployment.kind == DISAGGREGATED else "session"
    return f"{prefix}-{deployment.name}"


@dataclass
class Compiled:
    """What rows and deployments make.

    The tasks, the rows their cards are rendered from, and the tasks whose files could not
    be placed (why, by task id).
    """

    tasks: list[graph_mod.Task]
    rows: list[rows_mod.Row]
    unplaced: dict[str, str]


def compile_tasks(
    rows: Sequence[rows_mod.Row],
    deployments: Sequence[Deployment],
    *,
    queue: Mapping[str, vocab_mod.QueueItem] | None = None,
    target_rel: str = "",
    own_paths: Sequence[str] = (),
    allowed_paths: Sequence[str] = (),
    repo_root: str = "",
    baseline: bool = False,
    budget: int = graph_mod.MAX_ATTEMPTS,
    required_checks: Sequence[str] = (),
    survey: str = "",
    ci: Sequence[str] = (),
) -> Compiled:
    """The task list, by code: one task per row, and each deployment's own.

    ============================  =========================================================
    an ``operator`` row           one ``kernel`` task on its entry and test
    any other row                 one task of the row's kind
    an aggregated deployment      ``assemble-<name>``, then its session ``session-<name>``
    a disaggregated deployment    its ``state`` rows (``pd-state-<name>`` when the survey
                                  wrote none), then ``pd-ctx-<name>`` and ``pd-gen-<name>``
                                  judged by row checks with no checkpoint, then the one
                                  session, ``pd-hop-<name>``
    ============================  =========================================================

    A row with no deployment is the target's: the first deployment's assembly waits for
    it. A row whose check needs the assembled model (``needs: model``) is the other way
    round: it starts once its deployment's assembly is done and that deployment's session
    waits for it, as does every row that waits for such a row. The model's files are in
    its write scope: a defect it meets there is its to repair, and the session that
    follows measures what it left.

    **The first deployment's session waits for nothing but its assembly.** Its pass is
    the run's first milestone (``onboarded``), and the assembly's own check has already
    measured it on the commit that landed: the session's task finds that row and the ref
    is pinned. A row that needs the first deployment's assembled model therefore runs
    after that session, beside the further deployments. What it changes of the model is
    measured once more in the final pass: that is the price of the row, not of the
    milestone. Every further deployment's tasks start once the first one's session passed
    (``onboarded``), side by side; with ``baseline`` (the spec names an onboarded ref) no
    deployment waits for another, and the rows are the delta. A session waits for another
    deployment's only when it is judged relative to it. Nothing here names an engine
    feature: what a deployment needs beyond its form is a row the survey wrote.

    ``ci`` are the files the delivery requires for CI (the target's accuracy test, the
    directory of the test lists, the reference file of its anchor's row): known from the
    spec, so no survey has to think of them. One task (:data:`rows.DELIVERY_ROW`) writes
    them once the first deployment's session passed, beside the further deployments, and
    puts right every finding of form a landing recorded: the delivery's form holds
    nothing before ``onboarded``. They are this task's alone: a survey row that names one
    of them does not get it in its write scope, and its card says whose it is.
    """
    queue = queue or {}
    own = [p for p in own_paths if p]
    first = deployments[0].name if deployments and not baseline else ""
    # With an onboarded baseline the target stands: a deployment's own tasks hear it,
    # and what they may change of it. (The survey hears it in its own text.)
    standing = (
        ". The target is onboarded already (the spec's `onboarded` ref) and serves what "
        "it was onboarded with: change of it only what this deployment needs"
        if baseline
        else ""
    )
    by_name = {d.name: d for d in deployments}
    required = set(required_checks)
    tasks: list[graph_mod.Task] = []
    kept: list[rows_mod.Row] = []
    unplaced: dict[str, str] = {}
    task_of: dict[str, str] = {}
    taken: set[str] = set()

    def home(row: rows_mod.Row) -> str:
        """The deployment a row's work is for: ``""`` for the target itself."""
        name = row.deployment if row.deployment in by_name else ""
        return "" if deployments and name == deployments[0].name else name

    def after_onboarded(name: str) -> list[str]:
        return [session_task_id(by_name[first])] if first and name and name != first else []

    def optional(name: str) -> bool:
        owed = set(by_name[name].checks) if name in by_name else set()
        return bool(owed) and bool(required) and not (owed & required)

    # The rows that come after an assembly, each with the deployment whose model it needs:
    # those that say so, and every row that waits for one of them.
    late: dict[str, str] = {}
    for row in rows:
        model = row.deployment if row.deployment in by_name else ""
        model = model or (deployments[0].name if deployments else "")
        if row.needs == rows_mod.NEEDS_MODEL and model:
            late[row.id] = model
    while True:
        behind = {
            row.id: next(late[d] for d in row.depends_on if d in late)
            for row in rows
            if row.id not in late and any(d in late for d in row.depends_on)
        }
        if not behind:
            break
        late.update(behind)

    made: list[tuple[rows_mod.Row, graph_mod.Task]] = []
    for row in rows:
        touches: list[str] = []
        routes: list[str] = []
        theirs: list[str] = []
        if row.kind == rows_mod.OPERATOR:
            item = queue.get(row.id) or vocab_mod.queue_item_from_ref(
                str(row.config_axes.get("entry") or ""), row.id, row.outcome
            )
            task_id = (
                _kernel_task_id(item, taken) if item is not None and row.id in queue else row.id
            )
            kind = "kernel"
            paths = vocab_mod.entry_paths(item, target_rel, repo_root or None) if item else None
            if item is None:
                unplaced[task_id] = "its row names no entry to write or certify"
            elif paths is None:
                unplaced[task_id] = (
                    f"its entry `{item.target}` cannot be placed: under a target the run does "
                    "not know"
                )
            else:
                # Its files are its own whether or not the run may commit them yet: a
                # resolution that widens the policy leaves the row with all of them.
                touches = list(paths)
                outside = [p for p in paths if not graph_mod.inside(p, allowed_paths)]
                if outside:
                    unplaced[task_id] = (
                        f"its entry `{item.target}` is outside the paths the run may commit "
                        f"(`commit_policy.allowed_paths`): {', '.join(outside)}"
                    )
        else:
            task_id, kind = row.id, row.kind
            # Files of its own, never a whole directory: what it states, else one module
            # and one test named for it. Its own check runs its test files, so it has one.
            # Not the files the delivery requires for CI: a run's row named the accuracy
            # test, and its own check then ran a four-device test on none.
            theirs = [w for w in row.writes if ci and graph_mod.inside(w, ci)]
            touches = [w for w in row.writes if w not in theirs] or rows_mod.own_files(row.id, own)
            if row.kind == rows_mod.ROUTING and target_rel:
                # Routing is the family's, not the target's: the file the delivery reads,
                # and for a family the models' package does not import yet the line that
                # registers it. A row that names no file gets these for its module.
                routes = list(delivery_rules.routing_places(repo_root or "", target_rel))
                named = list(row.writes) or [t for t in touches if rows_mod.test_files([t])]
                touches = list(
                    dict.fromkeys([*named, *routes] if row.writes else [*routes, *named])
                )
            if row.check.startswith(f"{rows_mod.CPU_CHECK}:") and not rows_mod.test_files(touches):
                touches += [t for t in (rows_mod.own_test(row.id, own),) if t]
            # Its own files against what the run may commit, now and not at its landing: a
            # whole run's row did its work and was refused there, with its attempts spent.
            outside = [p for p in touches if not graph_mod.inside(p, allowed_paths)]
            if outside:
                unplaced[task_id] = (
                    "its files are outside the paths the run may commit "
                    f"(`commit_policy.allowed_paths`): {', '.join(outside)}"
                )
        goal = row.outcome
        if theirs:
            goal = (
                f"{goal.rstrip('.')}. {', '.join(f'`{p}`' for p in theirs)} is not in your write "
                f"scope: the task `{rows_mod.DELIVERY_ROW}` writes what the delivery requires "
                "for CI, after `onboarded`"
            )
        if routes:
            goal = (
                f"{goal.rstrip('.')}. The engine finds this target through its family's "
                f"routing, `{routes[0]}`"
                + (
                    f", and a family is registered by its one line in `{routes[1]}`: the "
                    "models' package does not import this family yet"
                    if len(routes) > 1
                    else ""
                )
                + ". Your write scope holds "
                + ("both" if len(routes) > 1 else "it")
            )
        # Where the delivery looks for the test of each entry among the row's files: in its
        # write scope from the start, and said on its card. A whole run's task put an
        # entry's cases where the delivery wants them and was refused by its own scope,
        # which named another test file: two rules of the flow, and no way to obey both.
        places = {
            path: found
            for path in (touches if kind != "kernel" and target_rel else ())
            if (found := delivery_rules.entry_test_places(repo_root or "", target_rel, path))
        }
        if places:
            touches = list(dict.fromkeys([*touches, *(p for f in places.values() for p in f)]))
            goal = (
                f"{goal.rstrip('.')}. The delivery looks for an entry's test where CI matches "
                "it, and that place is in your write scope: "
                + "; ".join(f"`{path}` is tested in `{found[0]}`" for path, found in places.items())
                + " (a multi-rank entry's op matrix and its rank body lie beside that name)"
            )
        if row.needs == rows_mod.NEEDS_MODEL and row.id in late:
            # Its check runs the assembled model: the model's files are in its scope, so
            # that a defect it finds there does not wait for somebody else's turn.
            touches = list(dict.fromkeys([*touches, *own]))
            goal = (
                f"{row.outcome.rstrip('.')}. Your check runs the assembled model of "
                f"`{late[row.id]}`, which is done: the target's own files are in your write "
                "scope. A defect in them that keeps your check from passing is yours to "
                "repair, with the smallest change that does it; say what you changed there "
                "in your receipt"
            )
        taken.add(task_id)
        task_of[row.id] = task_id
        task = graph_mod.Task(
            id=task_id,
            kind=kind,
            goal=goal,
            touches=touches,
            priority="nice_to_have" if optional(home(row)) else "required",
            budget=budget,
            ref=row.id,
            check=row.check,
        )
        tasks.append(task)
        kept.append(row)
        made.append((row, task))
    for row, task in made:
        deps = [task_of[d] for d in row.depends_on if d in task_of]
        task.depends_on = list(dict.fromkeys([*deps, *after_onboarded(home(row))]))

    # The survey's rows, and only they: the rows made below for the deployments themselves
    # are no row another deployment waits for (and have no task among these).
    surveyed = [row for row, _ in made]

    def needed(name: str) -> list[rows_mod.Row]:
        """The rows a deployment's own tasks wait for: its own, and the target's for the first."""
        shared = bool(deployments) and (baseline or name == deployments[0].name)
        return [
            r
            for r in surveyed
            if r.id not in late and (home(r) == name or (shared and home(r) == ""))
        ]

    def after(name: str) -> list[graph_mod.Task]:
        """The tasks of the rows that come after ``name``'s assembly, in the rows' order."""
        return [task for row, task in made if late.get(row.id) == name]

    for deployment in deployments:
        name = deployment.name
        onboard = after_onboarded(name)
        session = session_task_id(deployment)
        size = dict(
            gpus=deployment.gpus,
            nodes=deployment.nodes,
            wall=deployment.wall or rows_mod.DEFAULT_WALL_S,
        )
        relative = (
            [session_task_id(by_name[deployment.relative_to])]
            if deployment.relative_to in by_name and deployment.relative_to != name
            else []
        )
        priority = "nice_to_have" if optional(name) else "required"
        mine = needed(name)
        if deployment.kind == AGGREGATED:
            row = rows_mod.deployment_row(
                name,
                row_id=f"assemble-{name}",
                outcome=(
                    f"the target serves deployment `{name}` with its native options: its "
                    "model classes, weights and routing are written, and its session passes"
                    if name == first
                    else f"the target, which already serves the first deployment, also serves "
                    f"`{name}` with its native options, and every deployment that passed "
                    "still does"
                ),
                required=(
                    "the engine routes this checkpoint and these options to the target, and "
                    "every rank says so",
                    "every weight of the checkpoint is loaded or named as intentionally skipped",
                    f"the deployment's session passes: `{rows_mod.DEPLOYMENT_CHECK}:{name}`",
                ),
                leads=(
                    "Which calls of the forward does each landed row's entry serve, and what "
                    "do their receipts say is not certified?",
                    "How does the engine choose a target for a checkpoint and options, and "
                    "what does a target declare to be chosen?",
                ),
                writes=own,
                **size,
            )
            kept.append(row)
            tasks.append(
                graph_mod.Task(
                    id=row.id,
                    kind="assemble",
                    goal=row.outcome,
                    depends_on=list(dict.fromkeys([task_of[r.id] for r in mine] + onboard)),
                    touches=list(own),
                    priority=priority,
                    budget=budget,
                    ref=row.id,
                    check=row.check,
                )
            )
            before = [row.id]
        else:
            states = [task_of[r.id] for r in mine if r.kind == rows_mod.STATE]
            if not states:
                # What shows the transfer before the session: the row's own test, which it
                # writes. The native tests the deployment names are judged once, by the
                # contract's check of them on the candidate the session measures.
                state_test = [t for t in (rows_mod.own_test(f"pd-state-{name}", own),) if t]
                state = rows_mod.deployment_row(
                    name,
                    kind=rows_mod.STATE,
                    row_id=f"pd-state-{name}",
                    check=f"{rows_mod.CPU_CHECK}:pd-state-{name}",
                    outcome=(
                        f"every per-sequence state object of the model is transferable from the "
                        f"context side of `{name}` to its generation side, whose layouts may "
                        "differ; no transport and no proxy is written" + standing
                    ),
                    required=(
                        "every state object a sequence carries is listed with the manager that owns it",
                        "sentinel values written on the context layout arrive exactly on the "
                        "generation layout, layer by layer",
                        "the generation side's first step reads the cached length the context left",
                        "the target owns no state the engine's managers cannot see",
                        "its own test shows that transfer in one process",
                    ),
                    leads=(
                        "Which manager declares each kind of per-sequence state, and how is it "
                        "sharded or replicated across ranks?",
                        "What converts state between two layouts of unequal width, and how are "
                        "layers matched across them?",
                        "Which existing test moves this kind of state between unequal layouts "
                        "in one process?",
                    ),
                    writes=[*own, *state_test],
                    gpus=1,
                )
                kept.append(state)
                tasks.append(
                    graph_mod.Task(
                        id=state.id,
                        kind="state",
                        goal=state.outcome,
                        depends_on=list(onboard),
                        touches=[*own, *state_test],
                        priority=priority,
                        budget=budget,
                        ref=state.id,
                        check=state.check,
                    )
                )
                states = [state.id]
            others = [task_of[r.id] for r in mine if r.kind != rows_mod.STATE]
            before = []
            for side, what in (("ctx", "context"), ("gen", "generation")):
                # Judged by its own test, on one node's devices at most: each side passes
                # before the other is written, and before the session's nodes are asked for.
                side_test = [t for t in (rows_mod.own_test(f"pd-{side}-{name}", own),) if t]
                row = rows_mod.deployment_row(
                    name,
                    side=side,
                    row_id=f"pd-{side}-{name}",
                    check=f"{rows_mod.CPU_CHECK}:pd-{side}-{name}",
                    outcome=(
                        f"the {what} side of `{name}` resolves its native options on the target "
                        "and owns its part of the state: judged by its row check, with no "
                        "checkpoint loaded" + standing
                    ),
                    required=(
                        f"the {what} side's native options resolve to this target",
                        f"its role's layout and the state it {'produces' if side == 'ctx' else 'receives'} "
                        "match what the state task proved",
                        "the cached length is what the other side expects",
                        "its own test shows its side in one process",
                    ),
                    leads=(
                        f"What does the engine ask of a model on the {what} side that an "
                        "aggregated server never asks?",
                        "Which options must the two sides agree on, and which may differ?",
                    ),
                    writes=[*own, *side_test],
                    gpus=max(1, deployment.row_gpus),
                )
                kept.append(row)
                tasks.append(
                    graph_mod.Task(
                        id=row.id,
                        kind="assemble",
                        goal=row.outcome,
                        depends_on=list(dict.fromkeys([*states, *others, *onboard])),
                        touches=[*own, *side_test],
                        priority=priority,
                        budget=budget,
                        ref=row.id,
                        check=row.check,
                    )
                )
                before.append(row.id)
        # What needs the assembled model starts once it is assembled, and the session,
        # which measures the model as those rows left it, waits for them. But not the
        # first deployment's: its session is the milestone and waits for its assembly
        # alone; those rows start once it passed, as every further deployment's tasks do.
        behind = after(name)
        milestone = bool(first) and name == first
        for task in behind:
            task.depends_on = list(
                dict.fromkeys([*task.depends_on, *before, *([session] if milestone else [])])
            )
        if milestone:
            behind = []
        tasks.append(
            graph_mod.Task(
                id=session,
                kind=graph_mod.CHECK_KIND,
                goal=(
                    f"the one session of deployment `{name}`: every role instance loaded once, "
                    "every check of it measured as a client"
                ),
                depends_on=list(dict.fromkeys([*before, *(t.id for t in behind), *relative])),
                satisfies=list(deployment.checks),
                priority=priority,
                budget=budget,
                ref=name,
                check=f"{rows_mod.DEPLOYMENT_CHECK}:{name}",
            )
        )
    if deployments and rows_mod.DELIVERY_ROW not in taken:
        # Its own alone: a run's survey had a row that owned the accuracy test, this task
        # then held none of the three, and the delivery found the list and the anchor's
        # row missing at the close.
        files = list(ci[:2])
        reference = ci[2] if len(ci) > 2 else ""
        row = rows_mod.Row(
            id=rows_mod.DELIVERY_ROW,
            kind=rows_mod.FEATURE,
            outcome=(
                "what the delivery requires of the target's files, beside its checks, is in "
                "place"
                + (
                    f": its accuracy test for CI at `{files[0]}`, named by a test list CI "
                    "reads; and every finding of form a landing recorded is put right"
                    if files
                    else ": every finding of form a landing recorded is put right"
                )
            ),
            required=(
                *(
                    (
                        f"`{files[0]}` exists: a test class built on the engine's accuracy "
                        "harness, one case per deployment the run scores, each stating a "
                        "limit of its own that fits a model load and the benchmark",
                        *(
                            f"a list under `{where}/` names it, by the file or a directory "
                            "above it, relative to the accuracy tests' root: CI runs no test "
                            "that no list names"
                            for where in files[1:]
                        ),
                        "it is not run here, and you do not load the model to try it: the "
                        "final tests import the file and list its cases, and each "
                        "deployment's session has measured the benchmark through the endpoint",
                    )
                    if files
                    else ()
                ),
                *(
                    (
                        f"`{reference}` has an entry under the model's key for the anchor "
                        "the run's benchmark is judged against: the accuracy test for CI "
                        "reads its anchor there. Where the spec declares the anchor, your "
                        "dispatch says the entry to add; you add that entry and change no "
                        "other",
                    )
                    if reference
                    else ()
                ),
                "each file your dispatch lists as breaking a rule of the delivery holds to "
                "that rule when you hand in: you change the file, never the rule, and what "
                "the file computes stays as its task's check measured it",
            ),
            leads=(
                *(
                    (
                        "Which class do the engine's accuracy tests subclass, and how does "
                        "the accuracy test of an existing target name its model, its options "
                        "and its benchmark?",
                        "Which test list holds the accuracy tests of targets like this one, "
                        "and in what form does it name a test?",
                    )
                    if files
                    else ()
                ),
                "For each file listed: what does the rule it breaks ask of such a file, and "
                "which file of the tree already holds to it?",
            ),
            writes=(*files, *((reference,) if reference else ()), *own),
            # No check of the gate's, so its card gives no command to run: what it must
            # show is in the lines above, and the landing asks it.
            check="",
        )
        kept.append(row)
        tasks.append(
            graph_mod.Task(
                id=row.id,
                kind=row.kind,
                goal=row.outcome,
                # After the first milestone, beside the further deployments: nothing of
                # the delivery's form stands before `onboarded`.
                depends_on=[session_task_id(by_name[first])] if first else [],
                touches=list(row.writes),
                budget=budget,
                ref=row.id,
                check=row.check,
            )
        )
        mine = [p for p in row.writes if p not in own]
        outside = [path for path in mine if not graph_mod.inside(path, allowed_paths)]
        if outside:
            unplaced[row.id] = (
                "the files the delivery requires for CI are outside the paths the run "
                f"may commit (`commit_policy.allowed_paths`): {', '.join(outside)}"
            )
    if survey:
        for task in tasks:
            if survey not in task.depends_on:
                task.depends_on.insert(0, survey)
    return Compiled(tasks, kept, unplaced)


def split_tasks(
    raw: Any,
    parent: graph_mod.Task,
    graph: graph_mod.Graph,
    *,
    workspace: Path | str,
    own_paths: Sequence[str] = (),
    allowed_paths: Sequence[str] = (),
    repo_root: str = "",
    budget: int = graph_mod.MAX_ATTEMPTS,
) -> tuple[list[graph_mod.Task], Callable[[], None]]:
    """The tasks of the typed rows a split names for ``parent``, and what keeps their rows.

    Each child starts where its parent could (after what the parent waited for) and has
    its priority; the parent then joins them (``escalation.apply_answer``), which calls the
    second value to write the rows to ``work.yaml`` once the graph they make is valid.
    """
    try:
        rows = rows_mod.split_rows(raw, [t.id for t in graph.tasks])
    except rows_mod.RowError as exc:
        raise escalation_mod.ResolveError(str(exc)) from exc
    survey = next((t for t in graph.tasks if t.kind == "survey"), None)
    compiled = compile_tasks(
        rows,
        [],
        target_rel=(survey.ref if survey is not None else "")
        or (own_paths[0] if own_paths else ""),
        own_paths=own_paths,
        allowed_paths=allowed_paths,
        repo_root=repo_root,
        baseline=True,
        budget=budget,
    )
    if compiled.unplaced:
        raise escalation_mod.ResolveError(
            "; ".join(f"{tid}: {why}" for tid, why in compiled.unplaced.items())
        )
    existing = {t.id for t in graph.tasks}
    asked = {row.id: [d for d in row.depends_on if d in existing] for row in rows}
    for task in compiled.tasks:
        # Every dependency it was given is kept: one on a task the graph already has is
        # an edge to that task, and the graph's validation refuses the split it makes
        # impossible (a child that waits for what waits for its parent).
        task.depends_on = list(
            dict.fromkeys([*parent.depends_on, *task.depends_on, *asked.get(task.ref, ())])
        )
        task.priority = parent.priority

    def keep() -> None:
        # An operator row the survey did not write gets a vocabulary of its own, where the
        # gate finds its entry and tests. Kept, with the rows, once the split is admissible.
        kept = [rows_mod.write_row_vocabulary(workspace, row) for row in compiled.rows]
        rows_mod.save_work(workspace, kept)

    return compiled.tasks, keep


def _kernel_task_id(item: vocab_mod.QueueItem, taken: Iterable[str]) -> str:
    """``k<row>-<entry name>`` for a queue row's kernel task, numbered on when that is taken."""
    taken = set(taken)
    stem = f"k{item.n}-{item.target.rstrip('/').rsplit('/', 1)[-1].replace('_', '-')}"
    name, n = stem, 1
    while name in taken:
        n += 1
        name = f"{stem}-{n}"
    return name


@contextlib.contextmanager
def stoppable(workspace: Path | str) -> Iterator[Callable[[str], None]]:
    """While open, the callable it yields stops the run this process drives on ``workspace``.

    Called with the reason, from any thread, it makes the loop start nothing
    more and return ``stopped: <reason>`` at once; the turns and checks in
    flight are left to end on their own and their outcomes are never read. A
    loop that starts after it was called stops as it starts. For what runs
    beside the run's first turns and can find the run cannot go on, such as
    preflight's allocation probe.
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


def stop_run(workspace: Path | str, reason: str) -> bool:
    """Stop the run this process drives on ``workspace``, asked from outside it.

    The same stop :func:`stoppable` yields, for a caller that is not inside the run (the watch
    on the controller's own time limit). False while no run is open there yet: the caller asks
    again, and a loop that starts after it took the stop stops as it starts.
    """
    with _STOPS_LOCK:
        stopped = _STOPS.get(Path(workspace).resolve())
        if stopped is None:
            return False
        if not stopped.done():
            stopped.set_result(reason)
        return True


__all__ = [
    "AGGREGATED",
    "BLOCKED",
    "CHECKED",
    "Compiled",
    "DEFAULT_MAX_PARALLEL_CHECKS",
    "DEFAULT_MAX_PARALLEL_TASKS",
    "DISAGGREGATED",
    "Deployment",
    "FINISHED",
    "GraphOrchestrator",
    "READY",
    "Services",
    "compile_tasks",
    "seed",
    "session_task_id",
    "split_tasks",
    "stop_run",
    "stoppable",
]
