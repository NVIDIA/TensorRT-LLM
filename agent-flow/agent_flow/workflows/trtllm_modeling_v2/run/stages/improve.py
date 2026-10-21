# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The improver: lessons, flow issues and patch drafts from what costs a run time.

It is woken by a worker that ended without finishing (every
``FAILED_RECEIPT`` among them), a kernel task that certifies again an entry
a task of this run already certified, a required task that has waited to start
for longer than :data:`READY_WAIT_S` once ready, a planner revision that left
the plan as it was, a flow issue the driver or a worker reported, and a
check whose verdict was void. Wakes are batched: the first waits
:attr:`ImproveStage.improver_batch_s` for others to join it, and one improver
turn at most is in flight, on a thread of its own, so no task, check or
planner turn waits for it. A wake is kept in ``improve/wakes.json`` until a
turn has read it, so a restart loses none. A wake whose signature
(:attr:`Wake.signature`) an open issue in ``flow_issues.yaml`` names as one
it answers opens no turn: it is recorded on that issue as a recurrence. The run's close is the last turn,
and the close waits for it.

A task's end wakes it from the task's thread (:meth:`ImproveStage._improver_task_ended`);
the rest come from a watcher that reads the workspace while the planned run
is under way (:meth:`ImproveStage._improver_watching`): ``graph.yaml``,
``contract_status.jsonl`` and ``flow_issues.reported.jsonl``.

A turn writes ``lessons.proposed.yaml``; :mod:`...agents.lessons` validates
the proposal and merges it into the store. Every turn of every role is then
shown the lessons whose scope matches it: all roles, its own role, this host,
this model family.

A lesson that works around the workflow's own code also files a flow issue
in ``flow_issues.yaml``, the operator's fix list, with the patch the improver
drafted when it applies to the code this run executes: a diff under
``improve/patches/`` that adds the failing scenario as a test. The operator
applies it; nothing here does. Once a lesson names the commit that fixed it
and the running code has that commit, it is retired: journaled once, and
shown to no worker. A lesson about a defect of the engine that an upstream
change fixed is a known defect: a later run on a tree without the fix is told
so at preflight (:meth:`ImproveStage._known_defects_warning`).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import socket
import sys
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import yaml

from ...agents import lessons as lessons_mod
from ...agents.roles import ROLES
from ...infra.execution.agent_gate import Priority, turn_priority
from ...infra.transport import ExternalAgentTransport
from ...onboard import vocabulary as vocab_mod
from ...spec.task_schema import KNOWLEDGE_MEASUREMENT, knowledge_mode
from .. import contract as contract_mod
from .. import graph as graph_mod
from ..orchestrator import BACKGROUND_KINDS, FINISHED
from ..progress import decision_value
from .planned import ROLE_BY_KIND, TASKS_DIRNAME

IMPROVER_ROLE = "improver"
#: How long a ready task waits to start before the improver hears of it.
READY_WAIT_S = 30 * 60.0
#: How long the first wake waits for others to join its improver turn.
WAKE_BATCH_S = 60.0
#: How often the watcher reads the workspace.
WATCH_S = 60.0
#: The wakes no improver turn has read yet, the wakes seen, and how far the watcher read.
WAKES_FILENAME = "wakes.json"
#: What each kind of wake is called in the improver's request.
WAKE_KINDS: dict[str, str] = {
    "task_failed": "a task ended without finishing",
    "recertification": "an entry is certified again",
    "ready_waiting": "a ready task waited to start",
    "planner_noop": "a planner revision changed nothing",
    "flow_issue": "a flow issue was reported",
    "check_void": "a check's verdict was void",
}
#: What a read of ``graph.yaml`` may raise while it is absent or being written.
_GRAPH_ERRORS = (OSError, yaml.YAMLError, AttributeError, TypeError, ValueError)


def host_scope_name() -> str:
    """This host's name without its node number: the nodes of one cluster share one scope."""
    name = socket.gethostname().split(".")[0].lower()
    return re.sub(r"[-_]?\d+$", "", name) or name


@dataclass(frozen=True)
class Wake:
    """One thing the improver is woken for."""

    #: One of :data:`WAKE_KINDS`.
    kind: str
    #: What makes it the same wake again: a key seen once does not wake the improver twice.
    key: str
    #: What happened, as the improver's request says it.
    detail: str
    #: When it was raised (``time.time()``).
    at: float = 0.0
    #: What makes it the same problem again, across keys: a flow issue that
    #: names it answers every later wake that has it. Its kind when empty.
    signature: str = ""

    def sign(self) -> str:
        """Its :attr:`signature`, or its kind."""
        return self.signature or self.kind


# ------------------------------------------------------------------ what wakes it


def ready_waits(
    graph: graph_mod.Graph, since: dict[str, float], now: float, threshold: float
) -> list[Wake]:
    """The required ready tasks that have waited ``threshold`` seconds to start, as wakes.

    A campaign (:data:`~..orchestrator.BACKGROUND_KINDS`) waits for the
    core's checks by design, and a ``nice_to_have`` task for devices no
    required task wants, so neither counts. ``since`` holds when each ready
    task (by id and attempt) was first seen ready, and is kept up to date
    here: a task no longer ready leaves it. Each wake says what may hold the
    task back: the devices it asks, the tasks running, and those of them it
    shares a path with.
    """
    running = [t for t in graph.tasks if t.status == "running"]
    ready = {
        f"{t.id}#{t.attempts}": t
        for t in graph_mod.ready(graph)
        if t.priority == "required" and t.kind not in BACKGROUND_KINDS
    }
    for key in set(since) - set(ready):
        del since[key]
    wakes = []
    for key, task in ready.items():
        waited = now - since.setdefault(key, now)
        if waited < threshold:
            continue
        why = [f"it asks {task.gpus} GPU(s)", f"{len(running)} task(s) are running"]
        shared = [t.id for t in running if graph_mod.touches_conflict(task, t)]
        if shared:
            why.append(f"it shares paths with {', '.join(f'`{t}`' for t in shared)}")
        wakes.append(
            Wake(
                "ready_waiting",
                f"ready:{key}",
                f"task `{task.id}` ({task.kind}, attempt {task.attempts}) has been ready to "
                f"start for {waited / 60:.0f} min and has not started: " + "; ".join(why),
                now,
            )
        )
    return wakes


def same_certification(one: str, other: str) -> bool:
    """Whether two ``<entry>[@<site>]`` certify the same thing.

    Rows on one shared entry at two named sites land in local variants of
    their own, each with its own cells, so neither certifies the other's
    again. A side that names no site may run any of the entry's cells.
    """
    entry, _, site = one.partition("@")
    other_entry, _, other_site = other.partition("@")
    return entry == other_entry and (not site or not other_site or site == other_site)


def recertifications(
    graph: graph_mod.Graph, entry_of: Callable[[graph_mod.Task], str]
) -> list[Wake]:
    """The open kernel tasks on an entry another kernel task of this run certified, as wakes.

    ``entry_of`` gives a task's ``<entry>[@<site>]``; :func:`same_certification` matches them.
    """
    kernels = [t for t in graph.tasks if t.kind == "kernel"]
    certified: list[tuple[str, str]] = []
    for task in kernels:
        entry = entry_of(task) if task.status == "done" else ""
        if entry:
            certified.append((entry, task.id))
    wakes = []
    for task in kernels:
        entry = entry_of(task) if task.status in ("pending", "running") else ""
        before = [
            tid
            for done, tid in certified
            if entry and tid != task.id and same_certification(entry, done)
        ]
        if before:
            wakes.append(
                Wake(
                    "recertification",
                    f"recert:{task.id}",
                    f"kernel task `{task.id}` certifies `{entry}` again, which "
                    f"{', '.join(f'`{t}`' for t in before)} certified in this run: "
                    f"{task.goal[:300]}",
                )
            )
    return wakes


def plan_signature(graph: graph_mod.Graph) -> str:
    """What a planner revision changes of ``graph``: its tasks as written, not where they stand.

    The statuses the orchestrator gives (running, done, failed) are left
    out; a retry still shows, as its attempt, and so does a task abandoned.
    """
    rows = []
    for task in sorted(graph.tasks, key=lambda t: t.id):
        row = task.as_dict()
        row["abandoned"] = row.pop("status") == "abandoned"
        rows.append(row)
    return hashlib.sha1(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def void_wakes(records: Iterable[Mapping[str, Any]]) -> list[Wake]:
    """The void verdicts among ``contract_status.jsonl`` records, as wakes.

    One per check and cause: the cause with its commits and numbers left
    out, so the same check voided again for the same reason wakes no one.
    """
    wakes = []
    for record in records:
        void = str(record.get("void") or "")
        if not void:
            continue
        check = f"{record.get('unit')}:{record.get('gate')}@{record.get('member')}"
        cause = re.sub(r"[0-9a-f]{7,64}|\d+", "#", void.lower())
        digest = hashlib.sha1(cause.encode()).hexdigest()[:10]
        wakes.append(
            Wake(
                "check_void",
                f"void:{check}:{digest}",
                f"check `{check}` measured nothing, its verdict void: {void[:400]}",
                signature=f"check_void:{digest}",
            )
        )
    return wakes


def reported_wakes(records: Sequence[Mapping[str, Any]], start: int) -> list[Wake]:
    """The flow issues reported from line ``start`` on (0-based), as wakes."""
    wakes = []
    for index, record in enumerate(records, start):
        said = f"{record.get('by') or 'someone'} reported"
        said += f" at {record['at']}" if record.get("at") else ""
        parts = [f"{said}: {record.get('symptom') or '(no symptom)'}"]
        parts += [
            f"{k}: {record[k]}" for k in ("impact", "suspected", "proposed_fix") if record.get(k)
        ]
        symptom = re.sub(r"[0-9a-f]{7,64}|\d+", "#", str(record.get("symptom") or "").lower())
        evidence = record.get("evidence") or []
        if evidence:
            listed = evidence if isinstance(evidence, list) else [evidence]
            parts.append("evidence: " + ", ".join(f"`{e}`" for e in listed))
        wakes.append(
            Wake(
                "flow_issue",
                f"reported:{index}",
                f"line {index + 1} of `{lessons_mod.REPORTED_ISSUES_FILENAME}`: "
                + "; ".join(parts),
                signature="flow_issue:" + hashlib.sha1(symptom.encode()).hexdigest()[:10],
            )
        )
    return wakes


class _Watch:
    """What the watcher remembers between its reads of the workspace, in this process."""

    def __init__(self) -> None:
        #: When each ready task, by id and attempt, was first seen ready.
        self.ready_since: dict[str, float] = {}
        self.revisions: int | None = None
        self.signature = ""

    def planner_noops(self, graph: graph_mod.Graph) -> list[Wake]:
        """The planner revisions since the last read that left the plan as it was, as wakes."""
        count, signature = len(graph.revisions), plan_signature(graph)
        before, self.revisions = self.revisions, count
        unchanged, self.signature = signature == self.signature, signature
        if before is None or count <= before or not unchanged:
            return []
        return [
            Wake(
                "planner_noop",
                f"noop:{revision.number}",
                f"planner revision {revision.number} left the plan as it was: "
                f"{revision.reason[:300]}",
            )
            for revision in graph.revisions[before:]
            if not revision.reason.startswith("orchestrator:")
        ]


class _WakeState:
    """``improve/wakes.json``: wakes no turn read, keys seen, files read, the last refusals."""

    def __init__(self, path: Path) -> None:
        self.path = path
        data: dict[str, Any] = {}
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8") or "{}")
            except (OSError, ValueError):
                data = {}
        self.pending = [Wake(**w) for w in data.get("pending") or () if isinstance(w, dict)]
        self.seen = set(data.get("seen") or ())
        #: Per workspace file the watcher reads, how many of its lines it has read.
        self.read: dict[str, int] = {str(k): int(v) for k, v in (data.get("read") or {}).items()}
        #: What Python refused of the last proposal, for the next turn to correct.
        self.refused = [str(r) for r in data.get("refused") or ()]

    def save(self) -> None:
        data = {
            "pending": [w.__dict__ for w in self.pending],
            "seen": sorted(self.seen),
            "read": self.read,
            "refused": self.refused,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        partial = self.path.with_name(self.path.name + ".partial")
        partial.write_text(json.dumps(data, indent=1), encoding="utf-8")
        os.replace(partial, self.path)


class ImproveStage:
    """A stage of :class:`~..workflow.TrtllmModelingV2Workflow`; see the module docstring."""

    #: One improver turn at a time: turns share the proposal file.
    _improve_lock = threading.Lock()
    _retired_lock = threading.Lock()
    _wake_lock = threading.Lock()
    #: The lessons directory, or None when lessons are off. Set by the workflow.
    lessons_dir: Path | None = None
    #: How long the first wake waits for others, how often the watcher reads the
    #: workspace, and how long a ready task waits before it wakes the improver.
    improver_batch_s: float = WAKE_BATCH_S
    improver_watch_s: float = WATCH_S
    improver_ready_wait_s: float = READY_WAIT_S

    def _lesson_store(self) -> lessons_mod.LessonStore | None:
        return lessons_mod.LessonStore(self.lessons_dir) if self.lessons_dir else None

    def _lesson_scope(self) -> tuple[str, str]:
        """The host and family names this run's lessons are scoped by."""
        family = str(getattr(getattr(self, "_preflight", None), "family", "") or "")
        return host_scope_name(), re.sub(r"[^A-Za-z0-9_.\-]", "-", family)

    def _improves_in_run(self) -> bool:
        """Whether the improver takes turns while the run works, or only once it closes.

        A run that measures the workflow (``knowledge: measurement``) is the
        flow's developers' own: its improver turns on what woke it as the run
        goes. A production run is the user's: its wakes wait, recorded, for
        the one turn at the close (:meth:`_improve`), which takes no agent slot
        from the work it would have been improving.
        """
        return self._measuring()

    def _measuring(self) -> bool:
        """Whether the run measures the workflow on a model it has not met (``knowledge``).

        Such a run is shown neither the engine-defect registry nor the
        lessons of the model's family; every other lesson is the workflow's.
        """
        return knowledge_mode(getattr(self, "task_data", None) or {}) == KNOWLEDGE_MEASUREMENT

    def _lessons_block(self, role: str) -> str:
        """The lessons a turn of ``role`` is shown; the improver reads the store itself."""
        store = self._lesson_store()
        if store is None or role == IMPROVER_ROLE:
            return ""
        host, family = self._lesson_scope()
        self._journal_retired(store.retired(host=host, family=family))
        shown = store.for_dispatch(role, host=host, family=family)
        if self._measuring():
            shown = [
                lesson
                for lesson in shown
                if lesson.kind != "engine_defect" and not lesson.scope.startswith("family:")
            ]
        return lessons_mod.render_block(shown)

    def _journal_retired(self, retired: list[lessons_mod.Lesson]) -> None:
        """Journal each retired lesson the first time this process meets it."""
        with self._retired_lock:
            reported = self.__dict__.setdefault("_retired_reported", set())
            new = [lesson for lesson in retired if lesson.id not in reported]
            reported.update(lesson.id for lesson in new)
        if new:
            self.journal.event(
                "lessons retired",
                "\n".join(
                    f"[{lesson.scope}] {lesson.trigger[:80]} (fixed by {lesson.fixed_by})"
                    for lesson in new
                ),
            )

    def _known_defects_warning(self) -> str:
        """The known engine defects whose upstream fix the tree under test lacks; ``""`` for none.

        Read from the lessons in the run's scopes (``engine_defect`` lessons,
        :func:`~...agents.lessons.known_defects`) and journaled; preflight
        adds it to the host facts every dispatch reads.
        """
        store = self._lesson_store()
        repo = str((getattr(self, "task_data", None) or {}).get("trtllm_repo_path") or "")
        if store is None or not repo or self._measuring():
            return ""
        host, family = self._lesson_scope()
        found = lessons_mod.known_defects(store, Path(repo), host=host, family=family)
        lines = [
            f"- `{d.lesson.id}` [{d.lesson.scope}] {d.lesson.trigger} ("
            + (
                f"fixed upstream in `{d.lesson.upstream_fix}`"
                if d.lesson.upstream_fix
                else f"open: {d.lesson.disposition}"
            )
            + ("" if d.state == "absent" else "; git could not tell whether this tree has it")
            + f"): {d.lesson.lesson}"
            for d in found
            if d.state != "present"
        ]
        if not lines:
            return ""
        text = "\n".join(
            [
                f"Known engine defects whose upstream fix `{repo}` may not have; when the "
                "symptom appears, apply the fix rather than diagnosing it again:",
                *lines,
            ]
        )
        self.journal.event("known engine defects", text)
        return text

    def _known_defect_hits(self, rows: list[dict]) -> list[lessons_mod.DefectHit]:
        """The vocabulary queue ``rows`` in a known engine defect's regime whose fix the tree lacks.

        None in a measurement run, or without the lessons store.
        """
        store = self._lesson_store()
        repo = str((getattr(self, "task_data", None) or {}).get("trtllm_repo_path") or "")
        if store is None or not repo or self._measuring():
            return []
        host, family = self._lesson_scope()
        defects = [
            d.lesson
            for d in lessons_mod.known_defects(store, Path(repo), host=host, family=family)
            if d.state != "present"
        ]
        return lessons_mod.match_defects(defects, rows)

    def _flow_code_line(self) -> str:
        """Where the workflow's code runs from, for checking a fix is in it."""
        code = lessons_mod.FLOW_CODE
        head = lessons_mod.RUNNING_HEAD
        if not head:
            return (
                f"The workflow's code runs from `{code}`, which is not a git checkout here: "
                "no lesson can be retired by a fix."
            )
        return (
            f"The workflow's code runs from `{code}` at commit `{head}`. A commit is in "
            f"the running code when `git -C {code} merge-base --is-ancestor <commit> {head}` "
            "succeeds."
        )

    def _patch_block(self) -> str:
        """Where a patch draft is made and checked, against the code this run executes."""
        layout = lessons_mod.flow_layout()
        if layout is None:
            return (
                "## Patch drafts\n\nThe workflow's code is not a git checkout here: draft no "
                "patch; the issue's `proposed_fix` says the fix."
            )
        tree = lessons_mod.FLOW_CODE.parents[2].resolve()
        tree_rel = os.path.relpath(tree, layout.top)
        tests = os.path.relpath(layout.top / layout.tests, tree)
        scratch = self.workspace / lessons_mod.IMPROVE_DIRNAME / "scratch" / "<name>"
        archived = "" if tree_rel == "." else f" {tree_rel}"
        work = scratch if tree_rel == "." else scratch / tree_rel
        return "\n".join(
            [
                "## Patch drafts",
                "",
                f"The workflow is the package `{layout.package}` and its tests `{layout.tests}` "
                f"in the checkout `{layout.top}`, at commit `{layout.head}`; the run executes "
                "that checkout, so never change it.",
                f"- copy: `mkdir -p {scratch} && git -C {layout.top} archive {layout.head}"
                f"{archived} | tar -x -C {scratch}`, then commit the copy as its base",
                f"- test: `cd {work} && PYTHONPATH=$PWD {sys.executable} -m pytest -q "
                f"{tests}/<test file> -p no:cacheprovider`, failing before the fix and passing "
                "after it",
                f"- diff: `git -C {scratch} add -A && git -C {scratch} diff --cached > "
                f"{self.workspace / lessons_mod.PATCHES_DIR}/<name>.diff`, named in the issue "
                f"as `patch: {lessons_mod.PATCHES_DIR}/<name>.diff`",
                f"- check, before you end your turn: `{lessons_mod.check_command(self.workspace)}`",
            ]
        )

    # ------------------------------------------------------------------ wakes
    def _wake_state(self) -> _WakeState:
        """This workspace's wake state; under :attr:`_wake_lock`."""
        state = self.__dict__.get("_improver_wakes")
        if state is None:
            path = self.workspace / lessons_mod.IMPROVE_DIRNAME / WAKES_FILENAME
            state = self.__dict__.setdefault("_improver_wakes", _WakeState(path))
        return state

    def _improver_wake(self, *wakes: Wake) -> list[Wake]:
        """Wake the improver for each of ``wakes`` whose key it has not seen; returns those.

        They wait for the next improver turn, which starts on a thread of its
        own once the first of them is :attr:`improver_batch_s` old, unless a
        turn is in flight: its thread takes them next. In a production run they
        wait for the turn at the close (:meth:`_improves_in_run`). One whose signature an
        open flow issue answers is recorded on that issue instead
        (:func:`~...agents.lessons.record_recurrences`), and wakes no turn.
        """
        if self._lesson_store() is None:
            return []
        start = None
        with self._wake_lock:
            state = self._wake_state()
            new = []
            for wake in wakes:
                if wake.key not in state.seen:
                    state.seen.add(wake.key)
                    new.append(wake if wake.at else replace(wake, at=time.time()))
            if not new:
                return []
            known = lessons_mod.record_recurrences(
                self.workspace,
                [
                    (
                        w.sign(),
                        f"{time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(w.at))} "
                        f"{w.detail[:300]}",
                    )
                    for w in new
                ],
            )
            if known:
                self.journal.event(
                    "improver wake answered",
                    f"{len(known)} wake(s) an open flow issue answers: "
                    + ", ".join(sorted(set(known))),
                )
                new = [w for w in new if w.sign() not in known]
            if not new:
                state.save()
                return []
            state.pending += new
            state.save()
            if self.__dict__.get("_improver_thread") is None and self._improves_in_run():
                start = threading.Thread(
                    target=self._improver_loop, name="mv2-improver", daemon=True
                )
                self._improver_thread = start
        if start is not None:
            start.start()
        return new

    def _improver_resume(self) -> None:
        """Start a turn on what woke the improver in an earlier process and no turn read."""
        if self._lesson_store() is None:
            return
        start = None
        with self._wake_lock:
            if (
                self._wake_state().pending
                and self.__dict__.get("_improver_thread") is None
                and self._improves_in_run()
            ):
                start = threading.Thread(
                    target=self._improver_loop, name="mv2-improver", daemon=True
                )
                self._improver_thread = start
        if start is not None:
            start.start()

    def _improver_loop(self) -> None:
        """Turn after turn on the pending wakes, until none is left; then the thread ends."""
        while True:
            with self._wake_lock:
                pending = self._wake_state().pending
                if not pending:
                    self._improver_thread = None
                    return
                delay = min(w.at for w in pending) + self.improver_batch_s - time.time()
            if delay > 0:
                time.sleep(delay)
                continue
            with self._improve_lock:
                with self._wake_lock:
                    wakes = list(self._wake_state().pending)
                if not wakes:
                    continue
                try:
                    self._improve_turn(wakes)
                except Exception as exc:  # an improver turn that raised costs the run nothing
                    self.journal.event(
                        "improver turn raised",
                        f"{type(exc).__name__}: {exc}; its {len(wakes)} wake(s) are dropped",
                    )
                self._wakes_read(wakes)

    def _wakes_read(self, wakes: Sequence[Wake]) -> None:
        keys = {w.key for w in wakes}
        with self._wake_lock:
            state = self._wake_state()
            state.pending = [w for w in state.pending if w.key not in keys]
            state.save()

    def _wakes_improver(self, task: graph_mod.Task, decision: str) -> bool:
        """A task's end the improver should read: an entry not certified, a stuck worker, a repeat.

        A smith's ``FAILED_RECEIPT`` always is: a receipt that failed is a
        certification whose cost a lesson can save the next time. So is a
        worker that ended with no decision, was blocked, or could not be
        integrated, and the same work failing again: the task itself on a
        later attempt (``attempts``, kept in the graph across a restart), or
        another task of its lineage (:func:`~..graph.lineage`, a task the
        planner wrote again under a new id) earlier in this process. Another
        first failure is the work itself, which the planner handles; a
        review's REJECT on its first attempt is the review doing its job. Two
        failures of one kind are not a repeat: two kernels, or an audit and a
        later review, fail on their own.
        """
        head = decision.partition("\n")[0]
        if head in FINISHED:
            return False
        if task.kind == "review" and head == "REJECT" and task.attempts < 2:
            return False
        key = graph_mod.lineage(task)
        with self._wake_lock:
            failed = self.__dict__.setdefault("_failed_lineages", set())
            repeated = task.attempts >= 2 or key in failed
            failed.add(key)
        return repeated or head in ("", "BLOCKED", "INTEGRATION_FAILED", "FAILED_RECEIPT")

    def _improver_task_ended(self, task: graph_mod.Task, decision: str) -> None:
        """Wake the improver for a task's end worth a lesson (:meth:`_wakes_improver`); never waits."""
        if self._lesson_store() is None or not self._wakes_improver(task, decision):
            return
        envelope = self.workspace / TASKS_DIRNAME / f"{task.id}.json"
        role = ROLE_BY_KIND.get(task.kind, task.kind)
        head = decision.partition("\n")[0]
        self._improver_wake(
            Wake(
                "task_failed",
                f"task:{task.id}:{task.attempts}:{head}",
                f"task `{task.id}` ({task.kind}, role {role}) ended "
                f"{decision or 'with no decision'}. Its envelope is `{envelope}`.",
                signature=f"task_failed:{graph_mod.lineage(task)}:{head}",
            )
        )

    # ---------------------------------------------------------------- watcher
    @contextlib.contextmanager
    def _improver_watching(self) -> Iterator[None]:
        """While open, a thread reads the workspace every :attr:`improver_watch_s` and wakes the improver.

        It starts with a turn on what an earlier process left unread. Without
        a lessons store it does nothing.
        """
        if self._lesson_store() is None:
            yield
            return
        self._improver_resume()
        stop = threading.Event()

        def watch() -> None:
            while True:
                try:
                    self._improver_tick()
                except _GRAPH_ERRORS as exc:  # a read that failed costs one look
                    self.journal.event("improver watch", f"{type(exc).__name__}: {exc}")
                if stop.wait(self.improver_watch_s):
                    return

        thread = threading.Thread(target=watch, name="mv2-improver-watch", daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join(timeout=30)

    def _improver_tick(self, now: float | None = None) -> list[Wake]:
        """Read the workspace once and wake the improver for what it shows; returns the new wakes."""
        if self._lesson_store() is None:
            return []
        now = time.time() if now is None else now
        watch = self.__dict__.setdefault("_improver_watch", _Watch())
        found: list[Wake] = []
        path = self.workspace / graph_mod.GRAPH_FILENAME
        if path.is_file():
            graph = graph_mod.Graph.load(path)
            found += ready_waits(graph, watch.ready_since, now, self.improver_ready_wait_s)
            found += recertifications(graph, self._entry_of())
            found += watch.planner_noops(graph)
        read: dict[str, int] = {}
        for name, wakes_of in (
            (contract_mod.STATUS_FILENAME, lambda records, start: void_wakes(records)),
            (lessons_mod.REPORTED_ISSUES_FILENAME, reported_wakes),
        ):
            records = lessons_mod.jsonl_records(self.workspace / name)
            with self._wake_lock:
                start = self._wake_state().read.get(name, 0)
            start = start if start <= len(records) else 0
            found += wakes_of(records[start:], start)
            read[name] = len(records)
        new = self._improver_wake(*(replace(w, at=now) for w in found))
        with self._wake_lock:
            state = self._wake_state()
            if read != {name: state.read.get(name, 0) for name in read}:
                state.read.update(read)
                state.save()
        return new

    def _entry_of(self) -> Callable[[graph_mod.Task], str]:
        """What a kernel task certifies, ``<entry>[@<site>]``: its ``ref``'s, or its queue row's.

        A ``promote:`` task certifies nothing: it merges the landed variants'
        cells into the catalog and adds none, so it is ``""`` however many
        entries it names.
        """
        rows: dict[str, str] = {}
        path = getattr(self, "vocabulary_path", None)
        if path is not None and Path(path).is_file():
            try:
                vocabulary = vocab_mod.load_vocabulary(path)
            except (vocab_mod.VocabularyError, OSError, ValueError):
                vocabulary = None
            if vocabulary is not None:
                rows = {
                    str(item.n): f"{item.target}@{item.site}" if item.site else item.target
                    for item in vocabulary.queue
                }

        def entry(task: graph_mod.Task) -> str:
            mode, named, target = task.ref.partition(":")
            if not named:
                return rows.get(task.ref, "")
            if mode.strip() == vocab_mod.MODE_PROMOTE:
                return ""
            name, _, site = (part.strip() for part in target.partition("@"))
            return f"{name}@{site}" if site else name

        return entry

    # ------------------------------------------------------------------ turns
    def _improve(self, what: str) -> list[str]:
        """One improver turn on ``what`` and every wake no turn has read; returns its changes.

        In the caller's thread, after the turn in flight, if any. The run's
        close calls it, so the last turn reads all that is left.
        """
        if self._lesson_store() is None:
            return []
        with self._improve_lock:
            with self._wake_lock:
                wakes = list(self._wake_state().pending)
            changes = self._improve_turn(wakes, what)
            self._wakes_read(wakes)
            return changes

    def _improve_turn(self, wakes: Sequence[Wake], what: str = "") -> list[str]:
        """One improver turn on ``wakes`` and ``what``; returns the changes to the store and issues.

        Under :attr:`_improve_lock`. What Python refuses of its proposal is
        journaled and shown to the next turn. The turn is side work: it takes an
        agent slot only after the planner and the graph's tasks.
        """
        with turn_priority(Priority.SIDE):
            return self._improve_turn_body(wakes, what)

    def _improve_turn_body(self, wakes: Sequence[Wake], what: str) -> list[str]:
        store = self._lesson_store()
        if store is None:
            return []
        proposed = self.workspace / lessons_mod.PROPOSED_FILENAME
        carried_from, carried = self._carry_unread_proposal(store, proposed)
        proposed.unlink(missing_ok=True)
        issues = self.workspace / lessons_mod.FLOW_ISSUES_FILENAME
        reported = self.workspace / lessons_mod.REPORTED_ISSUES_FILENAME
        host, family = self._lesson_scope()
        known = [
            lesson
            for role in ROLES
            for lesson in store.for_dispatch(role, host=host, family=family)
        ]
        unique = {lesson.id: lesson for lesson in known}.values()
        retired = store.retired(host=host, family=family)
        with self._wake_lock:
            refused = list(self._wake_state().refused)
        if wakes:
            counts: dict[str, int] = {}
            for wake in wakes:
                counts[wake.kind] = counts.get(wake.kind, 0) + 1
            self.journal.event(
                "improver woken",
                ", ".join(f"{n} {WAKE_KINDS.get(k, k)}" for k, n in counts.items()),
            )
        if carried_from:
            what = "\n".join(
                [
                    *([what] if what else []),
                    f"Improver turn {carried_from} answered after its process stopped; its "
                    "proposal was merged before this turn: "
                    + ("; ".join(carried) or "nothing new"),
                ]
            )
        happened = ([what] if what else []) + [
            f"- **{WAKE_KINDS.get(w.kind, w.kind)}** ({time.strftime('%H:%M', time.localtime(w.at))}, "
            f"signature `{w.sign()}`): {w.detail}"
            for w in wakes
        ]
        prompt = "\n".join(
            [
                "## What happened",
                "",
                *happened,
                "",
                f"Workspace: `{self.workspace}`. The journal is `{self.journal_path}`, "
                f"the decision files are under `{self.workspace / 'decisions'}`, "
                f"the gate logs under `{self.workspace / 'gates'}`.",
                f"Host scope name: `host:{host}`."
                + (f" Family scope name: `family:{family}`." if family else ""),
                self._flow_code_line(),
                f"Flow issues filed in this run: `{issues}`"
                + ("." if issues.is_file() else " (none yet)."),
                f"Flow issues the driver and the workers reported: `{reported}`"
                + ("." if reported.is_file() else " (none yet)."),
                f"Write your proposal to `{proposed}`.",
                "",
                self._patch_block(),
                *(
                    [
                        "",
                        "## Your last proposal",
                        "",
                        "Python refused this of it; propose again what still holds, corrected:",
                        *[f"- {r}" for r in refused],
                    ]
                    if refused
                    else []
                ),
                "",
                lessons_mod.render_block(list(unique)) or "The store has no lessons in scope yet.",
                "",
                "## Retired lessons",
                "",
                "Their fix is in the running code; they stay in the store and no worker "
                "is shown them. Repeating one's trigger without `fixed_by` reopens it.",
                "",
                *(
                    [
                        f"- `{lesson.id}` [{lesson.scope}] {lesson.trigger} "
                        f"(fixed by `{lesson.fixed_by}`)"
                        for lesson in retired
                    ]
                    or ["None."]
                ),
            ]
        )
        decision = self._dispatch_task(IMPROVER_ROLE, "improve", "improve", prompt)
        if decision != "LESSONS_PROPOSED":
            return carried
        return carried + self._merge_proposal(store)

    def _carry_unread_proposal(self, store: Any, proposed: Path) -> tuple[str, list[str]]:
        """Merge the proposal of an improver turn answered after its process stopped.

        On the external transport a turn outlives the process that asked it;
        one answered between a stop and the next process's first improver
        turn left its proposal unread, and that turn would delete the file
        before asking again. So the answer is read here first, and its
        proposal merged, the way the turn's own process would have. Returns
        the turn's id (``""`` when there was none) and what the merge changed.
        """
        transport = getattr(self, "_turns", None)
        if not isinstance(transport, ExternalAgentTransport) or not proposed.is_file():
            return "", []
        unread = transport.unread_answer(IMPROVER_ROLE, "improve")
        if unread is None or decision_value(unread.decision_file, IMPROVER_ROLE) != (
            "LESSONS_PROPOSED"
        ):
            return "", []
        changes = self._merge_proposal(store)
        self.journal.event(
            "lessons carried over",
            f"{unread.request_id}: " + ("; ".join(changes) or "nothing new"),
        )
        return unread.request_id, changes

    def _merge_proposal(self, store: Any) -> list[str]:
        """Merge the workspace's proposed lessons and flow issues; what changed."""
        lessons, filed, errors = lessons_mod.load_proposal(self.workspace)
        with self._wake_lock:
            state = self._wake_state()
            state.refused = errors
            state.save()
        if errors:
            self.journal.event("lessons refused", "\n".join(errors))
        changes = store.merge(lessons, run=self.workspace.name, evidence_root=self.workspace)
        if changes:
            self.journal.event("lessons", "\n".join(changes))
        fixes = {lesson.id: lesson.fixed_by for lesson in lessons if lesson.fixed_by}
        recorded = lessons_mod.record_issues(self.workspace, filed, fixes)
        if recorded:
            self.journal.event("flow issues", "\n".join(recorded))
        return changes + recorded


__all__ = [
    "IMPROVER_ROLE",
    "ImproveStage",
    "READY_WAIT_S",
    "WAKE_BATCH_S",
    "WAKE_KINDS",
    "WATCH_S",
    "Wake",
    "host_scope_name",
    "plan_signature",
    "ready_waits",
    "recertifications",
    "reported_wakes",
    "same_certification",
    "void_wakes",
]
