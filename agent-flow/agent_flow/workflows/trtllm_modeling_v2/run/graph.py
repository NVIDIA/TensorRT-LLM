# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The task graph: the planner's output, validated and scheduled here.

Admissible means: known kinds, unique ids, dependencies that exist and form
no cycle, every required contract check claimed by some task, paths inside
what the commit policy allows, no task asking for more devices than the
site has.

Scheduling means: among tasks whose dependencies are done, first any task
that was running when the last process stopped, then required tasks on the
critical path, then other required tasks, then ``nice_to_have``
tasks only on devices no waiting required task could use; never two running
tasks that declare the same file, and nothing beside an ``exclusive`` task.

A soft dependency (``depends_on_soft``) lets a task start once the task it
names has started; it is held back only at its integration (:func:`ready`).
A ``check`` task's work is only contract checks, run with no agent turn.
A task touches only the run's own paths (:func:`own_paths`); code other models
run is not the run's to change, unless the operator lists it in
``commit_policy.allowed_paths``.
"""

from __future__ import annotations

import posixpath
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping, Sequence

import yaml

from ..infra.files import write_text_atomic
from ..operators.coverage import model_ops_tests_dir
from .workspace.integration import INTEGRATED_KINDS

#: A task whose work is only contract checks, run with no agent turn.
CHECK_KIND = "check"
KINDS: tuple[str, ...] = (
    "survey",
    "kernel",
    "assemble",
    "feature",
    "integrate",
    "investigate",
    CHECK_KIND,
)
#: What a worker's task can be tagged with: every kind an agent's turn serves. A
#: task's kind is its first tag; ``tags`` adds others, each briefing the worker
#: on one more kind of work (``agents/prompts/tags.py``).
WORKER_TAGS: tuple[str, ...] = tuple(k for k in KINDS if k != CHECK_KIND)
PRIORITIES: tuple[str, ...] = ("required", "nice_to_have")
#: ``split``: a kernel task that ended ``MACRO``, its work handed on to its constituents'
#: tasks; it lands nothing, as a failed one, and is not a failure.
STATUSES: tuple[str, ...] = ("pending", "running", "done", "failed", "abandoned", "split")
GRAPH_FILENAME = "graph.yaml"


@dataclass
class Task:
    id: str
    kind: str
    goal: str
    depends_on: list[str] = field(default_factory=list)
    #: Tasks this one may start beside once they have started: their change
    #: reaches it before its integration, which waits for them to land or end.
    depends_on_soft: list[str] = field(default_factory=list)
    touches: list[str] = field(default_factory=list)
    gpus: int = 0
    est_minutes: int = 60
    satisfies: list[str] = field(default_factory=list)
    priority: str = "required"
    status: str = "pending"
    #: Which run of this task is the latest: 1, plus one per retry of a failed run.
    attempts: int = 1
    #: Worker tags besides the kind, each a kind of work the task also does.
    tags: list[str] = field(default_factory=list)
    #: What the task works on, by kind: the unit (assemble, feature),
    #: the vocabulary queue row number (kernel).
    ref: str = ""
    #: The task runs alone: no other task runs beside it. For work that needs a
    #: stable tree, such as a native rebuild every worktree's build links to.
    exclusive: bool = False

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Task":
        resources = dict(data.get("resources") or {})
        return cls(
            id=str(data.get("id", "")),
            kind=str(data.get("kind", "")),
            goal=str(data.get("goal", "")),
            depends_on=[str(d) for d in data.get("depends_on") or []],
            depends_on_soft=[str(d) for d in data.get("depends_on_soft") or []],
            touches=[str(t) for t in data.get("touches") or []],
            gpus=int(resources.get("gpus", data.get("gpus", 0)) or 0),
            est_minutes=int(resources.get("est_minutes", data.get("est_minutes", 60)) or 60),
            satisfies=[str(s) for s in data.get("satisfies") or []],
            priority=str(data.get("priority", "required")),
            status=str(data.get("status", "pending")),
            attempts=int(data.get("attempts", 1) or 1),
            tags=[str(s) for s in data.get("tags") or []],
            ref=str(data.get("ref", "")),
            exclusive=_flag(data.get("exclusive", False)),
        )

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["resources"] = {"gpus": out.pop("gpus"), "est_minutes": out.pop("est_minutes")}
        if not out["tags"]:
            del out["tags"]
        return out


def _flag(value: Any) -> bool:
    return str(value).strip().lower() in ("true", "yes", "1")


@dataclass
class Revision:
    number: int
    reason: str


@dataclass
class Graph:
    tasks: list[Task] = field(default_factory=list)
    revisions: list[Revision] = field(default_factory=list)

    def by_id(self) -> dict[str, Task]:
        return {t.id: t for t in self.tasks}

    @classmethod
    def load(cls, path: Path) -> "Graph":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls(
            tasks=[Task.from_dict(t) for t in data.get("tasks") or []],
            revisions=[Revision(**r) for r in data.get("revisions") or []],
        )

    def save(self, path: Path) -> Path:
        """Write the graph to ``path`` whole: a reader never sees half of it."""
        data = {
            "tasks": [t.as_dict() for t in self.tasks],
            "revisions": [asdict(r) for r in self.revisions],
        }
        write_text_atomic(Path(path), yaml.safe_dump(data, sort_keys=False))
        return Path(path)


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

    Either is ``exclusive``, or both declare the very same file. Directories
    never conflict: each task commits on its own branch and is rebased onto the
    checkout when it finishes.
    """
    if a.exclusive or b.exclusive:
        return True
    return bool(_files(a) & _files(b))


#: Where a run's integration tests, their lists and the accuracy references live: data and
#: tests no other model's code runs.
INTEGRATION_TESTS_DIR = "tests/integration"


def own_paths(target_rel: str) -> tuple[str, ...]:
    """The run's own paths: its target's family, the tests that mirror it, the integration tests.

    The family's directory (its targets and the ``ops/`` they share) and its
    tests' directory, the two registries a new family adds a line to
    (``models/__init__.py``, ``_router_index.py``), and the integration tests
    (:data:`INTEGRATION_TESTS_DIR`: the accuracy test, its test list entry and
    its reference). Every other path is shared: code other models run, which a
    run does not change. None without a target.
    """
    if not target_rel.strip():
        return ()
    family = posixpath.dirname(posixpath.normpath(target_rel))
    models = posixpath.dirname(family)
    package = posixpath.dirname(models)
    return (
        family,
        model_ops_tests_dir(family),
        f"{models}/__init__.py",
        f"{package}/_router_index.py",
        INTEGRATION_TESTS_DIR,
    )


def builds_on(tasks: Mapping[str, Task], tid: str, roots: Iterable[str]) -> bool:
    """Whether ``tid`` reaches one of ``roots`` through its hard or soft dependencies."""
    wanted, seen = set(roots), set()
    stack = [tid]
    while stack:
        task = tasks.get(stack.pop())
        if task is None:
            continue
        for dep in (*task.depends_on, *task.depends_on_soft):
            if dep in wanted:
                return True
            if dep not in seen:
                seen.add(dep)
                stack.append(dep)
    return False


def _cycle(tasks: Mapping[str, Task]) -> list[str]:
    """One dependency cycle, as ids, or [] when there is none."""
    state: dict[str, int] = {}
    stack: list[str] = []

    def visit(tid: str) -> list[str]:
        state[tid] = 1
        stack.append(tid)
        for dep in [*tasks[tid].depends_on, *tasks[tid].depends_on_soft]:
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
    max_gpus: int | None = None,
    known_checks: Iterable[str] | None = None,
    own_paths: Sequence[str] = (),
) -> list[str]:
    """Every reason ``graph`` is not admissible, batched.

    ``known_checks`` are the contract's checks: a ``check`` task names only those, as it has nothing
    else to run. With ``own_paths`` (:func:`own_paths`) a task touches nothing outside them.
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
            errors.append(f"{where}: no goal")
        for dep in t.depends_on:
            if dep not in tasks:
                errors.append(f"{where}: depends on unknown task {dep!r}")
        errors += [f"{where}: {e}" for e in _soft_errors(t, tasks)]
        errors += [f"{where}: {e}" for e in _gating_errors(t, tasks)]
        if t.kind == CHECK_KIND:
            errors += [f"{where}: {e}" for e in _check_task_errors(t, known_checks)]
        unknown = [tag for tag in t.tags if tag not in WORKER_TAGS]
        if unknown:
            errors.append(f"{where}: tags {unknown} are not among {list(WORKER_TAGS)}")
        for path in t.touches:
            if not inside(path, allowed_paths):
                errors.append(f"{where}: touches {path!r}, outside {list(allowed_paths)}")
            elif own_paths and not inside(path, own_paths):
                errors.append(
                    f"{where}: touches {path!r}, code other models run, outside the run's own "
                    f"paths {list(own_paths)}: a shared kernel lands as a local variant under the "
                    "target's `ops/`; any other shared change is escalated, for a pull request "
                    "of its own"
                )
        if max_gpus is not None and t.gpus > max_gpus:
            errors.append(f"{where}: asks for {t.gpus} GPUs; the site has {max_gpus}")
    cycle = _cycle(tasks)
    if cycle:
        errors.append("dependency cycle: " + " -> ".join(cycle))
    claimed = {c for t in graph.tasks if t.status != "abandoned" for c in t.satisfies}
    for check in required_checks:
        if check not in claimed:
            errors.append(f"required check {check!r} is in no task's `satisfies`")
    return errors


def _gating_errors(task: Task, tasks: Mapping[str, Task]) -> list[str]:
    """Why ``task``'s hard dependencies would let an investigation or a check gate it.

    A task that owes contract checks (a ``check``, or one that ``satisfies`` some)
    never waits for an ``investigate`` task or a ``nice_to_have`` one: an
    investigation informs the checks, it does not gate them, and a strict
    diagnosis held as a precondition kept every benchmark of a run from running.
    A ``kernel`` task never waits for a
    ``check``: kernel certification runs on its own, whatever an integration
    check is doing.
    """
    errors = []
    owes = task.kind == CHECK_KIND or bool(task.satisfies)
    for dep in task.depends_on:
        other = tasks.get(dep)
        if other is None:
            continue
        if owes and (other.kind == "investigate" or other.priority == "nice_to_have"):
            what = "an `investigate`" if other.kind == "investigate" else "a nice_to_have"
            errors.append(
                f"depends on {dep!r}, {what} task: what it finds informs the checks and never "
                "gates them. Drop the edge (its findings reach the planner); a fix it calls for "
                "is a task of its own"
            )
        elif task.kind == "kernel" and other.kind == CHECK_KIND:
            errors.append(
                f"depends on {dep!r}, a `check` task: kernel certification never waits for an "
                "integration check; drop the edge"
            )
    return errors


def _soft_errors(task: Task, tasks: Mapping[str, Task]) -> list[str]:
    """Why ``task``'s soft dependencies are not admissible.

    A soft dependency holds its dependent back at the dependent's
    integration, until the change it names has landed: both ends are tasks
    whose commits land on the target.
    """
    errors = []
    if task.depends_on_soft and task.kind not in INTEGRATED_KINDS:
        errors.append(
            f"`depends_on_soft` holds a task back only at its integration, and a "
            f"{task.kind!r} task is not integrated: make it `depends_on`"
        )
    for dep in task.depends_on_soft:
        other = tasks.get(dep)
        if dep == task.id:
            errors.append("soft-depends on itself")
        elif other is None:
            errors.append(f"soft-depends on unknown task {dep!r}")
        elif dep in task.depends_on:
            errors.append(f"{dep!r} is in both `depends_on` and `depends_on_soft`: keep one")
        elif other.kind not in INTEGRATED_KINDS:
            errors.append(
                f"soft-depends on {dep!r}, a {other.kind!r} task, which lands nothing on the "
                "target: make it `depends_on`"
            )
    return errors


def _check_task_errors(task: Task, known_checks: Iterable[str] | None) -> list[str]:
    """Why a ``check`` task is not admissible: it runs contract checks and nothing else."""
    errors = []
    if not task.satisfies:
        errors.append("a `check` task runs the contract checks in its `satisfies`, and names none")
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
    if task.tags:
        errors.append("a `check` task runs no worker, so it takes no `tags`")
    if task.gpus:
        errors.append(
            "a `check` task declares `resources.gpus: 0`: each of its checks takes its own "
            "devices as it starts"
        )
    return errors


def critical_path_minutes(graph: Graph) -> dict[str, int]:
    """Per task, the longest remaining chain of estimated minutes through it to a sink.

    A soft dependency is on the chain too: its dependent's integration waits for it.
    """
    tasks = graph.by_id()
    dependents: dict[str, list[str]] = {t: [] for t in tasks}
    for t in graph.tasks:
        for dep in [*t.depends_on, *t.depends_on_soft]:
            if dep in dependents:
                dependents[dep].append(t.id)
    memo: dict[str, int] = {}

    def length(tid: str) -> int:
        if tid not in memo:
            task = tasks[tid]
            own = 0 if task.status == "done" else task.est_minutes
            memo[tid] = own + max((length(d) for d in dependents[tid]), default=0)
        return memo[tid]

    return {tid: length(tid) for tid in tasks}


def ready(graph: Graph, receipts_landed: Mapping[str, int] | None = None) -> list[Task]:
    """Pending tasks whose dependencies are all met and whose soft dependencies have started.

    A dependency is met once it is done, or once it failed with a receipt
    whose commits landed on the target: ``receipts_landed`` maps such a task
    to the attempt that landed them, and only its latest attempt counts.

    A soft dependency that is running, has landed or has ended without
    landing lets its dependent start; one still pending, or no longer in
    the graph, holds nothing back but the former.
    """
    tasks = graph.by_id()
    landed = receipts_landed or {}

    def met(dep: str) -> bool:
        task = tasks.get(dep)
        return task is not None and (
            task.status == "done" or (task.status == "failed" and landed.get(dep) == task.attempts)
        )

    return [
        t
        for t in graph.tasks
        if t.status == "pending"
        and all(met(d) for d in t.depends_on)
        and all(d not in tasks or tasks[d].status != "pending" for d in t.depends_on_soft)
    ]


def _id_order(tid: str) -> tuple[str | int, ...]:
    """``tid`` as a sort key whose runs of digits compare as numbers: ``k3-x`` before ``k10-x``."""
    return tuple(int(part) if i % 2 else part for i, part in enumerate(re.split(r"(\d+)", tid)))


def next_to_start(
    graph: Graph,
    free_gpus: int,
    resumed: Iterable[str] = (),
    held: Callable[[Task], int] | None = None,
    held_back: Iterable[str] = (),
    receipts_landed: Mapping[str, int] | None = None,
    waited: Mapping[str, float] | None = None,
) -> list[Task]:
    """The ready tasks to start now, given ``free_gpus`` devices to give out.

    ``receipts_landed`` are the failed tasks whose receipt's commits landed
    (:func:`ready`).

    ``held(task)`` is what a task takes of them while it runs (its ``gpus`` without it).
    ``held_back`` are ready tasks the caller keeps waiting; they claim nothing.

    Tasks in ``resumed``, those running when the last process stopped, go
    first; one that cannot start yet keeps its claim: nothing that conflicts
    with it starts before it does.

    Then required tasks, longest critical path first, a task's minutes waiting ready (``waited``) counting
    as path so a short one is not passed over for ever; on equal paths, in
    the order the ids number them (:func:`_id_order`). The caller starts only
    as many as it has slots, so this order decides which run. A
    ``nice_to_have`` task starts only while no ready required task is waiting
    for devices. No two running or starting tasks may conflict
    (:func:`touches_conflict`). An ``exclusive`` task that must wait for others to finish holds
    back every task ranked after it, or the run would never empty for it.
    """
    first = set(resumed)
    waiting = set(held_back)
    if _cycle(graph.by_id()):
        return []
    path = critical_path_minutes(graph)
    aged = dict(waited or {})
    running = [t for t in graph.tasks if t.status == "running"]
    candidates = sorted(
        ready(graph, receipts_landed),
        key=lambda t: (
            t.id not in first,
            t.priority != "required",
            -(path.get(t.id, 0) + aged.get(t.id, 0.0)),
            _id_order(t.id),
        ),
    )
    chosen: list[Task] = []
    # Resumed tasks that cannot start yet, still holding their files.
    claimed: list[Task] = []
    waiting_required = False
    for task in candidates:
        if task.id in waiting:
            continue
        if task.priority != "required" and waiting_required:
            continue
        if any(touches_conflict(task, other) for other in running + chosen + claimed):
            if task.exclusive:
                break
            if task.id in first:
                claimed.append(task)
            continue
        need = task.gpus if held is None else held(task)
        if need > free_gpus:
            if task.id in first:
                claimed.append(task)
            if task.priority == "required":
                waiting_required = True
            continue
        free_gpus -= need
        chosen.append(task)
    return chosen


#: Statuses only the orchestrator sets: it starts, finishes and fails tasks.
ORCHESTRATOR_STATUSES = frozenset({"running", "done", "failed", "split"})
#: How many worker runs one task id gets: the first, and each retry the planner asks for.
MAX_ATTEMPTS = 3
#: The longest goal a planner should write or rewrite: what to do and how it is
#: judged. Evidence, logs and the operator's messages stay in files the goal cites.
#: A longer one is adopted, and the planner is told (:func:`long_goals`).
MAX_GOAL_CHARS = 2000


def apply_revision(
    current: Graph,
    proposed: Graph,
    reason: str,
    *,
    not_started_attempts: Mapping[str, Sequence[int]] | None = None,
    max_attempts: int = MAX_ATTEMPTS,
    **validation: Any,
) -> tuple[Graph, list[str]]:
    """Adopt the planner's ``proposed`` graph when it is admissible.

    Statuses the orchestrator owns are carried over: a task that is running,
    done or failed stays so whatever the planner wrote, and a running task the
    proposal drops is kept. A planner that writes one of those three statuses for any
    other task gets ``pending``, as a task marked running that no worker runs
    never finishes.

    A planner may change ``failed`` to ``pending``: a retry, which runs the
    task again under its own id as its next attempt. Attempts are the
    orchestrator's count. A task whose ``max_attempts``-th worker run
    failed is not retried, and the revision is refused; an attempt in
    ``not_started_attempts`` (its worktree could not be prepared, so no worker
    ran) keeps its number but is not counted. Returns the adopted graph
    (``current`` on refusal) and the reasons for refusal.
    """
    if not reason.strip():
        return current, ["a revision must state its reason"]
    known = current.by_id()
    owned = {tid: t for tid, t in known.items() if t.status in ORCHESTRATOR_STATUSES}
    merged: list[Task] = []
    refused: list[str] = []
    for task in proposed.tasks:
        before = known.get(task.id)
        task.attempts = before.attempts if before is not None else 1
        if before is not None and before.status == "failed" and task.status == "pending":
            unstarted = set((not_started_attempts or {}).get(task.id, ()))
            if before.attempts - len(unstarted) >= max_attempts:
                refused.append(
                    f"task {task.id!r} failed on attempt {before.attempts} of {max_attempts} "
                    "and is not retried again: find the cause first (an `investigate` task), "
                    "reshape the work, or drop the task"
                )
            task.attempts += 1
        elif task.id in owned:
            task.status = owned[task.id].status
        elif task.status in ORCHESTRATOR_STATUSES:
            task.status = "pending"
        merged.append(task)
    kept = {t.id for t in merged}
    merged += [t for tid, t in owned.items() if t.status == "running" and tid not in kept]
    candidate = Graph(
        tasks=merged,
        revisions=[*current.revisions, Revision(len(current.revisions) + 1, reason.strip())],
    )
    errors = refused + validate(candidate, **validation)
    return (current, errors) if errors else (candidate, [])


def long_goals(current: Graph, proposed: Graph) -> list[str]:
    """The goals ``proposed`` writes or changes past :data:`MAX_GOAL_CHARS`, one note each."""
    known = current.by_id()
    return [
        f"task {task.id!r}: its goal is {len(task.goal)} characters, over {MAX_GOAL_CHARS}; "
        "it was adopted, but every worker turn of the task carries it: keep a goal to what to "
        "do and how it is judged, and cite evidence, logs and operator messages by the path of "
        "the file that holds them"
        for task in proposed.tasks
        if len(task.goal) > MAX_GOAL_CHARS
        and (task.id not in known or known[task.id].goal != task.goal)
    ]


def stalled(revisions_since_progress: int, cap: int) -> bool:
    """The planner has revised ``cap`` times without progress.

    Progress is a task done, a change landed on the target or a contract check
    turning green (``GraphOrchestrator._finish``).
    """
    return revisions_since_progress >= cap


__all__ = [
    "CHECK_KIND",
    "GRAPH_FILENAME",
    "Graph",
    "KINDS",
    "MAX_GOAL_CHARS",
    "Revision",
    "Task",
    "WORKER_TAGS",
    "apply_revision",
    "builds_on",
    "critical_path_minutes",
    "inside",
    "long_goals",
    "next_to_start",
    "own_paths",
    "ready",
    "stalled",
    "touches_conflict",
    "validate",
]
