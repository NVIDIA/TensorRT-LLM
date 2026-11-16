# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The task graph: made by code from the survey's rows and the spec's deployments, scheduled here.

Admissible means: known kinds, unique ids, dependencies that exist and form
no cycle, every required contract check claimed by some task, paths inside
what the commit policy allows.

Scheduling means: among tasks whose dependencies are done (or skipped),
first any task that was running when the last process stopped, then required
tasks, then ``nice_to_have`` ones, each in the graph's order; never two
running tasks that declare the same file. The caller starts as many as it has
agent turns; devices
are the pool's, never counted here.

A task is ``pending``, ``running``, ``done``, ``blocked`` (it waits for an
answer of the outer session) or ``skipped`` (reported not done). It has one attempt counter
and a budget of attempts. A ``check`` task's work is only contract checks,
run with no agent turn. Nothing here follows a change to shared code with a job:
the tests it reaches run once, in the final pass (``GraphOrchestrator._final_pass``).
"""

from __future__ import annotations

import posixpath
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import yaml

from ..infra.files import write_text_atomic
from ..operators.coverage import model_ops_tests_dir

#: A task whose work is only contract checks, run with no agent turn.
CHECK_KIND = "check"
#: What a task is, by what made it: the survey itself; a row of the survey by its kind (an
#: ``operator`` row is a ``kernel`` task); the assembly of one deployment (or one side of a
#: disaggregated one); and a ``check``.
KINDS: tuple[str, ...] = (
    "survey",
    "kernel",
    "weights",
    "routing",
    "state",
    "feature",
    "assemble",
    CHECK_KIND,
)
#: The kinds an agent's turn serves; each is briefed by its own section (``agents/prompts/tags.py``).
WORKER_KINDS: tuple[str, ...] = tuple(k for k in KINDS if k != CHECK_KIND)
PRIORITIES: tuple[str, ...] = ("required", "nice_to_have")
#: ``blocked``: its attempt could not go on, and its receipt asks the outer session
#: (``run/escalation.py``). ``skipped``: answered as not to be done, and reported so.
STATUSES: tuple[str, ...] = ("pending", "running", "done", "blocked", "skipped")
#: A dependency in one of these holds nothing back: a skipped one is said to its dependents.
MET: tuple[str, ...] = ("done", "skipped")
#: How many attempts a task gets unless the run says otherwise; ``resolve retry`` adds one.
MAX_ATTEMPTS = 3
GRAPH_FILENAME = "graph.yaml"


@dataclass
class Task:
    id: str
    kind: str
    #: The outcome: what done means.
    goal: str
    depends_on: list[str] = field(default_factory=list)
    #: Its write scope: the files and directories its commits may change.
    touches: list[str] = field(default_factory=list)
    #: The contract checks the run measures once its work has landed.
    satisfies: list[str] = field(default_factory=list)
    priority: str = "required"
    status: str = "pending"
    #: Which attempt is the latest: 1, plus one per attempt that did not end done.
    attempts: int = 1
    #: How many attempts it may use before it is blocked for the outer session.
    budget: int = MAX_ATTEMPTS
    #: What the task works on: the row of ``work.yaml`` its card is rendered from, or,
    #: for a deployment's session, the deployment.
    ref: str = ""
    #: Its one check, by registry id: ``kernel:<row>``, ``cpu:<row>`` or
    #: ``deployment:<name>``.
    check: str = ""
    #: How its previous attempt ended, for the next attempt's card.
    last: str = ""
    #: Of a blocked task: who blocked it (``by``), as what (``kind``), and what it asks of
    #: the outer session (``question``). Of a task that is done and still asks (a doubt its
    #: work did not wait for): the question, which holds nothing. Empty otherwise. What was observed is in ``last``;
    #: what its worker tried and ruled out is in its receipt, which is written once.
    block: dict[str, str] = field(default_factory=dict)
    #: Of a task that is done: each of its checks that failed nothing and could not observe
    #: what it is for, as ``<check>: <why>`` (the ledger's third outcome). The run closes
    #: ``partial`` on one, and repairs nothing for it.
    unverified: list[str] = field(default_factory=list)
    #: Of the first deployment's session: the commit ``onboarded`` was pinned on, once its
    #: publication succeeded. A link to the record (``onboarded.json``), never read in its
    #: place: it is how a start tells a publication that is owed from one that was made.
    onboarded: str = ""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Task":
        return cls(
            id=str(data.get("id", "")),
            kind=str(data.get("kind", "")),
            goal=str(data.get("goal", "")),
            depends_on=[str(d) for d in data.get("depends_on") or []],
            touches=[str(t) for t in data.get("touches") or []],
            satisfies=[str(s) for s in data.get("satisfies") or []],
            priority=str(data.get("priority", "required")),
            status=str(data.get("status", "pending")),
            attempts=int(data.get("attempts", 1) or 1),
            budget=int(data.get("budget", MAX_ATTEMPTS) or MAX_ATTEMPTS),
            ref=str(data.get("ref", "")),
            check=str(data.get("check", "")),
            last=str(data.get("last", "")),
            block={str(k): str(v) for k, v in (data.get("block") or {}).items()},
            unverified=[str(u) for u in data.get("unverified") or []],
            onboarded=str(data.get("onboarded") or ""),
        )

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        for key in ("check", "last", "block", "unverified", "onboarded"):
            if not out[key]:
                del out[key]
        return out


@dataclass
class Graph:
    tasks: list[Task] = field(default_factory=list)
    #: What the run itself asks of the outer session, when a final pass did not hold and
    #: no task owns the failure: its kind, what was observed and expected, the question,
    #: and when. Empty while the run asks nothing; an answer empties it.
    run_blocked: dict[str, str] = field(default_factory=dict)

    def by_id(self) -> dict[str, Task]:
        return {t.id: t for t in self.tasks}

    @classmethod
    def load(cls, path: Path) -> "Graph":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        tasks = [Task.from_dict(t) for t in data.get("tasks") or [] if isinstance(t, Mapping)]
        asked = data.get("run_blocked")
        return cls(
            tasks=tasks,
            run_blocked={str(k): str(v) for k, v in asked.items()}
            if isinstance(asked, Mapping)
            else {},
        )

    def save(self, path: Path) -> Path:
        """Write the graph to ``path`` whole: a reader never sees half of it."""
        data: dict[str, Any] = {"tasks": [t.as_dict() for t in self.tasks]}
        if self.run_blocked:
            data["run_blocked"] = dict(self.run_blocked)
        write_text_atomic(Path(path), yaml.safe_dump(data, sort_keys=False))
        return Path(path)


def removed_kinds(path: Path) -> list[str]:
    """What in a saved graph this flow does not have, one item each; empty for a graph of its own.

    A task of a kind or in a status that is gone (``id (kind)``, ``id (status)``), a
    regression check, or the planner's revisions: such a graph was written by an older
    flow, no code is left to run it, and the workspace is refused before anything is
    dispatched.
    """
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return []
    tasks = data.get("tasks") if isinstance(data, Mapping) else None
    found = [
        f"{task.get('id')} ({task.get('kind')})"
        for task in tasks or ()
        if isinstance(task, Mapping) and task.get("kind") not in KINDS
    ]
    found += [
        f"{task.get('id')} ({task.get('status')})"
        for task in tasks or ()
        if isinstance(task, Mapping)
        and task.get("kind") in KINDS
        and task.get("status", "pending") not in STATUSES
    ]
    found += [
        f"{task.get('id')} (a regression check)"
        for task in tasks or ()
        if isinstance(task, Mapping) and task.get("regression")
    ]
    if isinstance(data, Mapping) and data.get("revisions"):
        found.append("the planner's revisions")
    return found


def _files(task: Task) -> set[str]:
    """The files ``task`` declares, as opposed to its directories.

    A file is a path whose last part has a suffix (``modeling.py``); a path
    that ends in ``/``, or whose last part has none, is a directory.
    """
    files = set()
    for path in task.touches:
        if path.endswith("/"):
            continue
        clean = posixpath.normpath(path)
        if PurePosixPath(clean).suffix:
            files.add(clean)
    return files


def _within(path: str, root: str) -> bool:
    path, root = path.rstrip("/"), root.rstrip("/")
    return path == root or path.startswith(root + "/")


def inside(path: str, roots: Sequence[str]) -> bool:
    """Whether ``path`` is one of ``roots`` or lies in one; with no roots, any path is."""
    return not roots or any(_within(path, root) for root in roots)


def touches_conflict(a: Task, b: Task) -> bool:
    """Whether ``a`` and ``b`` may not run at the same time.

    What one may write overlaps what the other may: the same file, the same directory, or
    a file inside the other's directory. Two tasks that were both given a directory would
    each be told it is theirs, so they run one after the other; tasks that name their own
    files run side by side.
    """
    return any(_within(p, q) or _within(q, p) for p in a.touches for q in b.touches)


def own_paths(target_rel: str) -> tuple[str, ...]:
    """The run's own paths: its target's directory, and the one its tests mirror it in.

    Every other path a task may change is shared: code another model runs too.
    None without a target.
    """
    if not target_rel.strip():
        return ()
    target = posixpath.normpath(target_rel)
    return (target, model_ops_tests_dir(target))


def shared_touches(task: Task, own: Sequence[str]) -> list[str]:
    """The paths of ``task``'s ``touches`` outside ``own``, the run's own paths (:func:`own_paths`).

    A directory that holds an own path, such as the target's family, is outside it.
    """
    roots = [posixpath.normpath(o) for o in own]
    return [p for p in task.touches if not any(_within(posixpath.normpath(p), r) for r in roots)]


def _cycle(tasks: Mapping[str, Task]) -> list[str]:
    """One dependency cycle, as ids, or [] when there is none."""
    state: dict[str, int] = {}
    stack: list[str] = []

    def visit(tid: str) -> list[str]:
        state[tid] = 1
        stack.append(tid)
        for dep in tasks[tid].depends_on:
            if dep not in tasks:
                continue
            if state.get(dep) == 1:
                return stack[stack.index(dep) :] + [dep]
            if state.get(dep) is None:
                found = visit(dep)
                if found:
                    return found
        state[tid] = 2
        stack.pop()
        return []

    for tid in tasks:
        if state.get(tid) is None:
            found = visit(tid)
            if found:
                return found
    return []


def validate(
    graph: Graph,
    *,
    required_checks: Iterable[str] = (),
    allowed_paths: Sequence[str] = (),
    known_checks: Iterable[str] | None = None,
) -> list[str]:
    """Every reason ``graph`` is not admissible, batched.

    ``known_checks`` are the contract's checks: a ``check`` task names only those, as it has nothing
    else to run.
    """
    errors: list[str] = []
    ids = [t.id for t in graph.tasks]
    for tid in sorted({i for i in ids if ids.count(i) > 1}):
        errors.append(f"task id {tid!r} is used more than once")
    tasks = graph.by_id()
    for t in graph.tasks:
        where = f"task {t.id!r}"
        if not t.id:
            errors.append("a task has no id")
        if t.kind not in KINDS:
            errors.append(f"{where}: kind {t.kind!r} is not one of {list(KINDS)}")
        if t.priority not in PRIORITIES:
            errors.append(f"{where}: priority {t.priority!r} is not one of {list(PRIORITIES)}")
        if t.status not in STATUSES:
            errors.append(f"{where}: status {t.status!r} is not one of {list(STATUSES)}")
        if not t.goal.strip():
            errors.append(f"{where}: no outcome")
        for dep in t.depends_on:
            if dep not in tasks:
                errors.append(f"{where}: depends on unknown task {dep!r}")
        if t.kind == CHECK_KIND:
            errors += [f"{where}: {e}" for e in _check_task_errors(t, known_checks)]
        for path in t.touches:
            if not inside(path, allowed_paths):
                errors.append(f"{where}: touches {path!r}, outside {list(allowed_paths)}")
    cycle = _cycle(tasks)
    if cycle:
        errors.append("dependency cycle: " + " -> ".join(cycle))
    claimed = {c for t in graph.tasks for c in t.satisfies}
    for check in required_checks:
        if check not in claimed:
            errors.append(f"required check {check!r} is in no task's `satisfies`")
    return errors


def _check_task_errors(task: Task, known_checks: Iterable[str] | None) -> list[str]:
    """Why a ``check`` task is not admissible: it runs contract checks and nothing else.

    One with no check of its own is a join: it is done once the tasks it depends on are
    (what a split leaves of the task it replaced).
    """
    errors = []
    if known_checks is not None:
        known = set(known_checks)
        unknown = [key for key in task.satisfies if key not in known]
        if unknown:
            errors.append(
                f"a `check` task runs contract checks, and {', '.join(unknown)} "
                "is no check of the contract"
            )
    if task.touches:
        errors.append("a `check` task changes nothing: its `touches` stay empty")
    return errors


def met(dependency: Task | None) -> bool:
    """Whether a task that depends on ``dependency`` may start.

    Done is met. A skipped task that was only nice to have is met too: what depended on it
    goes on and is told so. A skipped **required** one is not: nothing produced what its
    dependents need, so they wait (and the run closes ``partial`` with them listed) until
    the session that watches the run resolves them as well. A blocked one holds its
    dependents until it is answered.
    """
    if dependency is None:
        return False
    return dependency.status == "done" or (
        dependency.status == "skipped" and dependency.priority != "required"
    )


def ready(graph: Graph) -> list[Task]:
    """Pending tasks whose every dependency is met (:func:`met`)."""
    tasks = graph.by_id()
    return [
        t
        for t in graph.tasks
        if t.status == "pending" and all(met(tasks.get(d)) for d in t.depends_on)
    ]


def blocked_by(graph: Graph, task_id: str) -> list[str]:
    """Every task that cannot start while ``task_id`` is not done: its dependents, transitively."""
    held: list[str] = []
    frontier = {task_id}
    while frontier:
        nxt = {
            t.id
            for t in graph.tasks
            if t.id not in held and t.id != task_id and frontier & set(t.depends_on)
        }
        held += [t.id for t in graph.tasks if t.id in nxt]
        frontier = nxt
    return held


def next_to_start(graph: Graph, resumed: Iterable[str] = ()) -> list[Task]:
    """The ready tasks that may start now, in the order they should.

    Tasks in ``resumed``, those running when the last process stopped, go
    first; one that cannot start yet keeps its claim: nothing that conflicts
    with it starts before it does.

    Then required tasks before ``nice_to_have`` ones, each in the graph's order (the
    survey's row order). The caller starts only as many as it has agent turns, so this
    order decides which run; devices are the pool's and are not counted here. No two
    running or starting tasks may conflict (:func:`touches_conflict`).
    """
    first = set(resumed)
    if _cycle(graph.by_id()):
        return []
    order = {t.id: i for i, t in enumerate(graph.tasks)}
    running = [t for t in graph.tasks if t.status == "running"]
    candidates = sorted(
        ready(graph),
        key=lambda t: (t.id not in first, t.priority != "required", order[t.id]),
    )
    chosen: list[Task] = []
    # Resumed tasks that cannot start yet, still holding their files.
    claimed: list[Task] = []
    for task in candidates:
        if any(touches_conflict(task, other) for other in running + chosen + claimed):
            if task.id in first:
                claimed.append(task)
            continue
        chosen.append(task)
    return chosen


__all__ = [
    "CHECK_KIND",
    "GRAPH_FILENAME",
    "Graph",
    "KINDS",
    "MAX_ATTEMPTS",
    "MET",
    "met",
    "STATUSES",
    "Task",
    "WORKER_KINDS",
    "blocked_by",
    "inside",
    "next_to_start",
    "own_paths",
    "ready",
    "removed_kinds",
    "shared_touches",
    "touches_conflict",
    "validate",
]
