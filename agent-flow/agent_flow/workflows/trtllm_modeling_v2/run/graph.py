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
tasks that declare the same file, and nothing beside an ``exclusive`` task;
and no task of a gated kind (``tune``, until the core's correctness checks
hold) at all.

A soft dependency (``depends_on_soft``) lets a task start once the task it
names has started; it is held back only at its integration (:func:`ready`).
A ``check`` task's work is only contract checks, run with no agent turn. A
regression check is a ``check`` task with ``regression: true``: it runs the
modeling_v2 package tests after the changes to shared code it covers
(:func:`regression_batch`).
"""

from __future__ import annotations

import copy
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
#: Task kinds that run for hours beside the graph on idle capacity: a performance
#: campaign. Its result is an artifact of the run, never a step another task builds
#: on, and it gives way to every other task (:func:`touches_conflict`).
BACKGROUND_KINDS = frozenset({"tune"})
KINDS: tuple[str, ...] = (
    "survey",
    "kernel",
    "assemble",
    "feature",
    "integrate",
    "investigate",
    "tune",
    CHECK_KIND,
)
#: What a worker's task can be tagged with: every kind an agent's turn serves. A
#: task's kind is its first tag; ``tags`` adds others, each briefing the worker
#: on one more kind of work (``agents/prompts/tags.py``).
WORKER_TAGS: tuple[str, ...] = tuple(k for k in KINDS if k != CHECK_KIND)
PRIORITIES: tuple[str, ...] = ("required", "nice_to_have")
STATUSES: tuple[str, ...] = ("pending", "running", "done", "failed", "abandoned")
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
    #: A ``check`` the orchestrator adds after changes to shared code: it runs the
    #: modeling_v2 package tests on them (:func:`add_regression_check`).
    regression: bool = False
    #: Of a regression check that has started, the changes to shared code it
    #: covers: those that had landed, uncovered, when it started
    #: (:func:`regression_batch`). The orchestrator's to write; ``None`` until then.
    covers: list[str] | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Task":
        resources = dict(data.get("resources") or {})
        covers = data.get("covers")
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
            regression=_flag(data.get("regression", False)),
            covers=None if covers is None else [str(c) for c in covers],
        )

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["resources"] = {"gpus": out.pop("gpus"), "est_minutes": out.pop("est_minutes")}
        if out["covers"] is None:
            del out["covers"]
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
    checkout when it finishes. A background campaign works on its own branches
    from the commit it pinned, so an ``exclusive`` task does not wait for it.
    """
    if (a.exclusive or b.exclusive) and not {a.kind, b.kind} & BACKGROUND_KINDS:
        return True
    return bool(_files(a) & _files(b))


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


def _ancestors(tasks: Mapping[str, Task], tid: str) -> set[str]:
    """Every task ``tid`` depends on, directly or through others."""
    seen: set[str] = set()
    stack = list(tasks[tid].depends_on)
    while stack:
        dep = stack.pop()
        if dep in tasks and dep not in seen:
            seen.add(dep)
            stack.extend(tasks[dep].depends_on)
    return seen


#: The statuses of a task whose change is on the target, or will be.
_LIVE = ("pending", "running", "done")


def is_regression_check(task: Task) -> bool:
    """Whether ``task`` is a required regression check that has not failed."""
    return (
        task.kind == CHECK_KIND
        and task.regression
        and task.priority == "required"
        and task.status in _LIVE
    )


def changes_shared_code(task: Task, own: Sequence[str]) -> bool:
    """Whether ``task``'s commits land on the target and its ``touches`` reach outside ``own``."""
    return task.kind in INTEGRATED_KINDS and bool(shared_touches(task, own))


def landing(task: Task) -> str:
    """How a regression check's batch names ``task``'s latest run: ``<id>@<attempt>``.

    A retry that lands again is a change no earlier check covered.
    """
    return f"{task.id}@{task.attempts}"


def regression_batch(
    graph: Graph, own: Sequence[str], landed: Mapping[str, int] | None = None
) -> list[str]:
    """The changes to shared code a regression check starting now would cover, as task ids.

    Every task that changes shared code (:func:`changes_shared_code`) and
    has landed, in graph order, whose landing no regression check that has
    started and not failed covers, whatever its ``depends_on``. A done task
    has landed, and so has any other whose ``landed`` attempt (task id to
    attempt, the orchestrator's record) is its latest.
    """
    landed = landed or {}
    covered: set[str] = set()
    for review in graph.tasks:
        if is_regression_check(review) and review.status != "pending":
            covered |= set(review.covers or ())
    return [
        t.id
        for t in graph.tasks
        if (t.status == "done" or landed.get(t.id) == t.attempts)
        and landing(t) not in covered
        and changes_shared_code(t, own)
    ]


def unreviewed_shared_changes(graph: Graph, own: Sequence[str]) -> dict[str, list[str]]:
    """Per task that changes shared code with no regression check after it, its shared paths.

    A regression check follows a task:

    - when it has not started and depends on the task, directly or through
      other tasks;
    - when the task is done and the check has started and covers its
      landing (its batch, :attr:`Task.covers`);
    - when the task is done and the check has not started: whatever its
      ``depends_on``, its batch takes every change landed before it starts.

    A failed task is not counted: whatever of it landed is in the batch of the
    next regression check (:func:`regression_batch`).
    """
    tasks = graph.by_id()
    followed: set[str] = set()
    covered: set[str] = set()
    waiting = False
    for review in graph.tasks:
        if not is_regression_check(review):
            continue
        if review.status == "pending":
            followed |= _ancestors(tasks, review.id)
            waiting = True
        else:
            covered |= set(review.covers or ())
    unreviewed: dict[str, list[str]] = {}
    for task in graph.tasks:
        if task.kind not in INTEGRATED_KINDS or task.status not in _LIVE:
            continue
        if task.id in followed:
            continue
        if task.status == "done" and (waiting or landing(task) in covered):
            continue
        shared = shared_touches(task, own)
        if shared:
            unreviewed[task.id] = shared
    return unreviewed


#: The id of the first regression check the orchestrator adds; later ones are numbered.
REGRESSION_CHECK_ID = "regression"


def add_regression_check(graph: Graph, own: Sequence[str]) -> Task | None:
    """Make every change to shared code followed by a regression check; the task added, or ``None``.

    The orchestrator's, not the planner's: a pending regression check takes,
    in its ``depends_on``, each task that changes shared code and has not
    landed yet; with none pending and a change no check follows, one is
    added. Without ``own`` paths nothing is shared, and nothing is added.
    After a regression check failed, nothing is added for what already landed:
    the same tree would fail the same way.
    """
    if not own:
        return None
    unreviewed = unreviewed_shared_changes(graph, own)
    if not unreviewed:
        return None
    tasks = graph.by_id()
    later = [tid for tid in unreviewed if tasks[tid].status != "done"]
    for check in graph.tasks:
        if is_regression_check(check) and check.status == "pending":
            check.depends_on += [tid for tid in later if tid not in check.depends_on]
            return None
    failed = any(
        t.kind == CHECK_KIND and t.regression and t.status == "failed" for t in graph.tasks
    )
    if failed and not later:
        return None
    number = 1 + sum(1 for t in graph.tasks if t.regression)
    tid = REGRESSION_CHECK_ID if number == 1 else f"{REGRESSION_CHECK_ID}-{number}"
    while tid in tasks:
        number += 1
        tid = f"{REGRESSION_CHECK_ID}-{number}"
    task = Task(
        id=tid,
        kind=CHECK_KIND,
        goal=(
            "run every modeling_v2 package test on the landed changes to shared code "
            "(source-level gates, the catalog's and the other targets' GPU tests), against the "
            "results at the base"
        ),
        depends_on=later,
        regression=True,
    )
    graph.tasks.append(task)
    return task


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
    required_kinds: Iterable[str] = (),
    known_checks: Iterable[str] | None = None,
) -> list[str]:
    """Every reason ``graph`` is not admissible, batched.

    ``required_kinds`` are kinds the run's goals need a task of, such as
    ``tune`` when the task asks for performance work. ``known_checks`` are the
    contract's checks: a ``check`` task names only those, as it has nothing
    else to run, unless it is a regression check.
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
        for dep in [*t.depends_on, *t.depends_on_soft]:
            if dep in tasks and tasks[dep].kind in BACKGROUND_KINDS:
                errors.append(
                    f"{where}: depends on {dep!r}, a `{tasks[dep].kind}` task; its result is "
                    "an artifact of the run that no task builds on or integrates"
                )
        errors += [f"{where}: {e}" for e in _soft_errors(t, tasks)]
        if t.kind == CHECK_KIND:
            errors += [f"{where}: {e}" for e in _check_task_errors(t, known_checks)]
        unknown = [tag for tag in t.tags if tag not in WORKER_TAGS]
        if unknown:
            errors.append(f"{where}: tags {unknown} are not among {list(WORKER_TAGS)}")
        for path in t.touches:
            if not inside(path, allowed_paths):
                errors.append(f"{where}: touches {path!r}, outside {list(allowed_paths)}")
        if max_gpus is not None and t.gpus > max_gpus:
            errors.append(f"{where}: asks for {t.gpus} GPUs; the site has {max_gpus}")
        if t.regression and t.kind != CHECK_KIND:
            errors.append(
                f"{where}: `regression: true` marks a regression check, and a {t.kind!r} task "
                "is not a `check`"
            )
    cycle = _cycle(tasks)
    if cycle:
        errors.append("dependency cycle: " + " -> ".join(cycle))
    claimed = {c for t in graph.tasks if t.status != "abandoned" for c in t.satisfies}
    for check in required_checks:
        if check not in claimed:
            errors.append(f"required check {check!r} is in no task's `satisfies`")
    kinds = {t.kind for t in graph.tasks if t.status != "abandoned"}
    for kind in required_kinds:
        if kind not in kinds:
            errors.append(f"the task asks for `{kind}` work and the graph has no `{kind}` task")
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
    if task.regression and task.satisfies:
        errors.append(
            "a regression check runs the package tests on its batch, not contract "
            "checks: its `satisfies` stays empty"
        )
    if not task.satisfies and not task.regression:
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
    gated_kinds: Iterable[str] = (),
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
    ``held_back`` are ready tasks the caller keeps waiting, such as a regression
    check whose batch is still open; they claim nothing.

    Tasks in ``resumed``, those running when the last process stopped, go
    first; one that cannot start yet keeps its claim: nothing that conflicts
    with it starts before it does.

    Then required tasks, a background campaign after the rest, longest
    critical path first, a task's minutes waiting ready (``waited``) counting
    as path so a short one is not passed over for ever; on equal paths, in
    the order the ids number them (:func:`_id_order`). The caller starts only
    as many as it has slots, so this order decides which run. A
    ``nice_to_have`` task starts only while no ready required task is waiting
    for devices. No two running or starting tasks may conflict
    (:func:`touches_conflict`), and no task of a kind in ``gated_kinds``
    starts. An ``exclusive`` task that must wait for others to finish holds
    back every task ranked after it, or the run would never empty for it.
    """
    gated = set(gated_kinds)
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
            t.kind in BACKGROUND_KINDS,
            -(path.get(t.id, 0) + aged.get(t.id, 0.0)),
            _id_order(t.id),
        ),
    )
    chosen: list[Task] = []
    # Resumed tasks that cannot start yet, still holding their files.
    claimed: list[Task] = []
    waiting_required = False
    for task in candidates:
        if task.kind in gated or task.id in waiting:
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
ORCHESTRATOR_STATUSES = frozenset({"running", "done", "failed"})
#: How many worker runs one task id gets: the first, and each retry the planner asks for.
MAX_ATTEMPTS = 3
#: The longest goal a planner may write or rewrite: what to do and how it is
#: judged. Evidence, logs and the operator's messages stay in files the goal cites.
MAX_GOAL_CHARS = 2000


def apply_revision(
    current: Graph,
    proposed: Graph,
    reason: str,
    *,
    not_started_attempts: Mapping[str, Sequence[int]] | None = None,
    **validation: Any,
) -> tuple[Graph, list[str]]:
    """Adopt the planner's ``proposed`` graph when it is admissible.

    Statuses the orchestrator owns are carried over: a task that is running,
    done or failed stays so whatever the planner wrote, and a running task the
    proposal drops is kept, and so is a started regression check's batch
    (:attr:`Task.covers`); a pending regression check the proposal leaves out
    is kept too. A planner that writes one of those three statuses for any
    other task gets ``pending``, as a task marked running that no worker runs
    never finishes.

    A planner may change ``failed`` to ``pending``: a retry, which runs the
    task again under its own id as its next attempt. Attempts are the
    orchestrator's count. A task whose :data:`MAX_ATTEMPTS`-th worker run
    failed is not retried, and the revision is refused; an attempt in
    ``not_started_attempts`` (its worktree could not be prepared, so no worker
    ran) keeps its number but is not counted. A goal the proposal writes or
    changes may be :data:`MAX_GOAL_CHARS` long at most. Returns the adopted
    graph (``current`` on refusal) and the reasons for refusal.
    """
    if not reason.strip():
        return current, ["a revision must state its reason"]
    known = current.by_id()
    owned = {tid: t for tid, t in known.items() if t.status in ORCHESTRATOR_STATUSES}
    merged: list[Task] = []
    refused: list[str] = []
    for task in proposed.tasks:
        before = known.get(task.id)
        if len(task.goal) > MAX_GOAL_CHARS and (before is None or before.goal != task.goal):
            refused.append(
                f"task {task.id!r}: its goal is {len(task.goal)} characters, over "
                f"{MAX_GOAL_CHARS}: say what to do and how it is judged, and cite evidence, "
                "logs and operator messages by the path of the file that holds them"
            )
        task.attempts = before.attempts if before is not None else 1
        if before is not None and before.status == "failed" and task.status == "pending":
            unstarted = set((not_started_attempts or {}).get(task.id, ()))
            if before.attempts - len(unstarted) >= MAX_ATTEMPTS:
                refused.append(
                    f"task {task.id!r} failed on attempt {before.attempts} of {MAX_ATTEMPTS} "
                    "and is not retried again: find the cause first (an `investigate` task), "
                    "reshape the work, or drop the task"
                )
            task.attempts += 1
        elif task.id in owned:
            task.status = owned[task.id].status
        elif task.status in ORCHESTRATOR_STATUSES:
            task.status = "pending"
        # What a regression check covered is recorded as it starts; a retry records anew.
        task.covers = before.covers if before is not None and task.status != "pending" else None
        merged.append(task)
    kept = {t.id for t in merged}
    merged += [t for tid, t in owned.items() if t.status == "running" and tid not in kept]
    merged += [
        copy.deepcopy(t)
        for t in current.tasks
        if is_regression_check(t) and t.status == "pending" and t.id not in kept
    ]
    ids = {t.id for t in merged}
    for task in merged:
        if is_regression_check(task):
            task.depends_on = [d for d in task.depends_on if d in ids]
    candidate = Graph(
        tasks=merged,
        revisions=[*current.revisions, Revision(len(current.revisions) + 1, reason.strip())],
    )
    errors = refused + validate(candidate, **validation)
    return (current, errors) if errors else (candidate, [])


def stalled(revisions_since_progress: int, cap: int) -> bool:
    """The planner has revised ``cap`` times without progress.

    Progress is a task done, a change landed on the target or a contract check
    turning green (``GraphOrchestrator._finish``).
    """
    return revisions_since_progress >= cap


__all__ = [
    "BACKGROUND_KINDS",
    "CHECK_KIND",
    "GRAPH_FILENAME",
    "Graph",
    "KINDS",
    "MAX_GOAL_CHARS",
    "REGRESSION_CHECK_ID",
    "Revision",
    "Task",
    "WORKER_TAGS",
    "add_regression_check",
    "apply_revision",
    "changes_shared_code",
    "critical_path_minutes",
    "inside",
    "is_regression_check",
    "landing",
    "next_to_start",
    "own_paths",
    "ready",
    "regression_batch",
    "shared_touches",
    "stalled",
    "touches_conflict",
    "unreviewed_shared_changes",
    "validate",
]
