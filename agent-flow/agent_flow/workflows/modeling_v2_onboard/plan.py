# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parse and validate the planner's ``plan.md``.

``plan.md`` carries one ``### <task-id>`` section per task (the prose a worker
reads) and a ``## Execution Graph`` block the scheduler runs. Each graph node is
a task; its ``type`` is the task's primary tag and an optional ``tags`` list adds
more. The engine's :class:`~agent_flow.orchestration.Node` keeps only ``type``,
so the tags are read from the same YAML block here.

The rules below are what make a plan runnable by this workflow. Every violation
is collected so the planner can fix them all in one retry.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import yaml

from agent_flow.orchestration import (
    ExecutionGraph,
    ExecutionGraphError,
    NodeState,
    parse_execution_graph,
)

__all__ = [
    "PRIORITIES",
    "TAGS",
    "PlanError",
    "PlanSpec",
    "parse_plan",
]

# The task tags a plan may use. Each one selects an extra block of worker
# system prompt (see ``prompts.tags``).
TAGS: tuple[str, ...] = (
    "new_op",
    "target",
    "merge",
    "module_accuracy",
    "e2e_accuracy",
    "perf",
)

# GPU priority classes, most urgent first: the path to the first whole-model
# accuracy result, then feature work, then performance work. The shared GPU
# pool serves a waiting task of an earlier class first.
PRIORITIES: tuple[str, ...] = ("accuracy", "feature", "perf")

# Task ids become git branch names (``node/<id>``) and file names, so keep them
# to a character set that is safe in both.
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

# Same heading shape ``agent_team.execution_graph`` accepts, including an
# optional leading section number.
_GRAPH_HEADING_RE = re.compile(r"##\s+(?:\d+[.)]?\s+)?Execution\s+Graph\s*$")
_FENCE = "```"


class PlanError(ValueError):
    """``plan.md`` is missing, malformed, or breaks a workflow rule.

    Attributes:
        problems: Every rule violation found, one human-readable line each.
    """

    def __init__(self, problems: Iterable[str]) -> None:
        self.problems = list(problems)
        super().__init__("plan.md is not runnable:\n- " + "\n- ".join(self.problems))


@dataclass(frozen=True)
class PlanSpec:
    """A validated plan.

    Attributes:
        graph: The execution graph the scheduler runs.
        tags: Each task's tags, primary tag first.
        text: The plan.md text this spec was parsed from.
        priorities: Each task's GPU priority class (one of :data:`PRIORITIES`).
            Not part of a task's definition: changing it never re-runs a task.
        side: The ``side`` tasks: ``perf`` work whose branch the run does not
            merge, listed for a human when the run ends. Not part of a task's
            definition either: a task moves in and out of it without re-running.
    """

    graph: ExecutionGraph
    tags: Mapping[str, tuple[str, ...]]
    text: str
    priorities: Mapping[str, str]
    side: frozenset[str] = frozenset()

    def sink_id(self) -> str:
        """The id of the plan's final task, the one every task but the side ones leads to."""
        return _sinks(self.graph, self.side)[0]

    def section(self, node_id: str) -> str | None:
        """The body of task ``node_id``'s ``###`` section."""
        return _section(self.text, node_id)

    def is_landing(self, node_id: str) -> bool:
        """Whether task ``node_id`` runs in the checkout and lands what it commits."""
        return self.graph.by_id[node_id].isolation == "shared"

    def entry(self, node_id: str) -> tuple[object, ...]:
        """Everything that defines task ``node_id``: its graph entry and its section."""
        node = self.graph.by_id[node_id]
        return (
            node.type,
            node.depends_on,
            node.kind,
            node.isolation,
            self.tags[node_id],
            self.section(node_id),
        )


def _graph_block(text: str) -> str | None:
    """Return the body of the first fenced block under ``## Execution Graph``."""
    lines = text.splitlines()
    start = next(
        (i for i, line in enumerate(lines) if _GRAPH_HEADING_RE.fullmatch(line.strip())),
        None,
    )
    if start is None:
        return None
    body: list[str] | None = None
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if body is None:
            if stripped.startswith("## "):
                return None
            if stripped.startswith(_FENCE):
                body = []
            continue
        if stripped == _FENCE:
            return "\n".join(body)
        body.append(line)
    return None


def _sinks(graph: ExecutionGraph, side: Iterable[str] = ()) -> list[str]:
    """The tasks outside ``side`` that no task outside ``side`` depends on."""
    side = set(side)
    main = [node for node in graph.nodes if node.id not in side]
    depended_on = {dep for node in main for dep in node.depends_on}
    return [node.id for node in main if node.id not in depended_on]


def _raw_field(block: str, key: str) -> dict[str, object]:
    data = yaml.safe_load(block)
    return {
        raw["id"]: raw.get(key)
        for raw in data["nodes"] or []
        if isinstance(raw, dict) and isinstance(raw.get("id"), str)
    }


def _ancestors(graph: ExecutionGraph, node_id: str) -> set[str]:
    seen: set[str] = set()
    stack = list(graph.by_id[node_id].depends_on)
    while stack:
        dep = stack.pop()
        if dep not in seen:
            seen.add(dep)
            stack.extend(graph.by_id[dep].depends_on)
    return seen


def default_priorities(
    graph: ExecutionGraph, tags: Mapping[str, tuple[str, ...]]
) -> dict[str, str]:
    """The priority class of each task when the plan does not set one.

    The gates are the ``e2e_accuracy`` tasks that build on a ``target`` task
    (without a ``target`` task, every ``e2e_accuracy`` task). The first gates
    are those that check the earliest code: no other gate is upstream of one,
    and no other gate's upstream tasks are a strict part of its own. Later
    gates need not depend on the first ones, since joins do not wait for
    checks. The first gates and everything upstream of them are ``accuracy``,
    ``perf``-tagged tasks among them included: they are on the way to the
    first result. The other ``perf`` tasks are ``perf``, and the rest are
    ``feature``.
    """
    ancestors = {node.id: _ancestors(graph, node.id) for node in graph.nodes}
    targets = {tid for tid, t in tags.items() if "target" in t}
    gates = {
        tid
        for tid, t in tags.items()
        if "e2e_accuracy" in t and (not targets or ancestors[tid] & targets)
    }
    first = {
        tid
        for tid in gates
        if not ancestors[tid] & gates
        and not any(ancestors[other] < ancestors[tid] for other in gates)
    }
    accuracy = first.union(*(ancestors[tid] for tid in first))
    return {
        node.id: "accuracy"
        if node.id in accuracy
        else "perf"
        if "perf" in tags[node.id]
        else "feature"
        for node in graph.nodes
    }


def _task_tags(node_type: str, extra: object, problems: list[str], node_id: str) -> tuple[str, ...]:
    if extra is None:
        extra = []
    if not isinstance(extra, list) or not all(isinstance(t, str) for t in extra):
        problems.append(f"task {node_id!r}: `tags` must be a list of strings")
        extra = []
    tags = tuple(dict.fromkeys([node_type, *extra]))
    unknown = [t for t in tags if t not in TAGS]
    if unknown:
        problems.append(
            f"task {node_id!r}: unknown tag(s) {unknown}; the tags are {', '.join(TAGS)}"
        )
    return tags


def _heading_re(node_id: str) -> re.Pattern[str]:
    # The id must end at whitespace or end of line, so `gsm8k` does not match
    # the heading of `gsm8k-final`.
    return re.compile(rf"^###\s+{re.escape(node_id)}(?=\s|$).*$", flags=re.MULTILINE)


def _has_task_section(text: str, node_id: str) -> bool:
    return _heading_re(node_id).search(text) is not None


def _section(text: str, node_id: str) -> str | None:
    """The body of task ``node_id``'s ``###`` section, up to the next heading."""
    heading = _heading_re(node_id).search(text)
    if heading is None:
        return None
    body = text[heading.end() :]
    end = re.search(r"^#{1,3}\s", body, flags=re.MULTILINE)
    return (body[: end.start()] if end else body).strip()


def _side_tasks(
    graph: ExecutionGraph,
    tags: Mapping[str, tuple[str, ...]],
    raw: Mapping[str, object],
    problems: list[str],
) -> frozenset[str]:
    """The tasks marked ``side: true``, checked against the rules for them."""
    side: set[str] = set()
    for node_id, value in raw.items():
        if node_id not in tags or value is None or value is False:
            continue
        if value is not True:
            problems.append(f"task {node_id!r}: `side` must be true or false, got {value!r}")
        elif "perf" not in tags[node_id] or "merge" in tags[node_id]:
            problems.append(
                f"task {node_id!r}: only a `perf` task that is not a `merge` task can be `side`"
            )
        elif graph.by_id[node_id].isolation == "shared":
            problems.append(f"task {node_id!r}: a `side` task cannot take `isolation: shared`")
        else:
            side.add(node_id)
    for node in graph.nodes:
        on_side = [dep for dep in node.depends_on if dep in side]
        if node.id not in side and on_side:
            problems.append(
                f"task {node.id!r} depends on side task(s) {on_side}; a side branch is never "
                "merged into the main line — drop `side` from a task to merge it"
            )
    return frozenset(side)


def _check_landing(
    graph: ExecutionGraph,
    tags: Mapping[str, tuple[str, ...]],
    side: frozenset[str],
    problems: list[str],
) -> None:
    """Check the final task and the chain of tasks that land in the checkout."""
    sinks = _sinks(graph, side)
    if len(sinks) != 1:
        problems.append(
            f"the graph must end in exactly one final task that every task but the "
            f"`side` ones leads to; found {len(sinks)}: {sinks}"
        )
    shared = [node.id for node in graph.nodes if node.isolation == "shared"]
    if len(sinks) == 1 and sinks[0] not in shared:
        problems.append(
            f"the final task {sinks[0]!r} must take `isolation: shared`, which lands the "
            f"result in the TensorRT-LLM checkout; found shared tasks {shared}"
        )
    for node_id in shared:
        if "merge" not in tags[node_id]:
            problems.append(f"the shared task {node_id!r} must be tagged `merge`")
    ancestors = {node_id: _ancestors(graph, node_id) for node_id in shared}
    chain = sorted(shared, key=lambda node_id: len(ancestors[node_id]))
    for earlier, later in zip(chain, chain[1:]):
        if earlier not in ancestors[later]:
            problems.append(
                f"shared tasks {earlier!r} and {later!r} would land side by side; each "
                "task that takes `isolation: shared` must depend, directly or not, on the "
                "one before it"
            )


def parse_plan(
    text: str,
    *,
    previous: PlanSpec | None = None,
    states: Mapping[str, NodeState] | None = None,
) -> PlanSpec:
    """Parse ``plan.md`` text and check every workflow rule.

    Args:
        text: The full ``plan.md``.
        previous: The plan currently running, when this is a revision.
        states: The current task states. A ``DONE`` or ``RUNNING`` task of
            ``previous`` must keep its graph entry and its section: finished
            work is never re-run, and a running worker is never re-briefed.

    Returns:
        The validated :class:`PlanSpec`.

    Raises:
        PlanError: Listing every rule the plan breaks.
    """
    block = _graph_block(text)
    if block is None:
        raise PlanError(['no "## Execution Graph" heading followed by a fenced ```yaml block'])
    try:
        graph = parse_execution_graph(block)
    except ExecutionGraphError as exc:
        raise PlanError([f"execution graph: {exc}"]) from exc

    problems: list[str] = []
    if not graph.nodes:
        raise PlanError(["the execution graph has no tasks"])

    raw_tags = _raw_field(block, "tags")
    raw_priorities = _raw_field(block, "priority")
    tags: dict[str, tuple[str, ...]] = {}
    for node in graph.nodes:
        if node.children:
            problems.append(
                f"task {node.id!r}: nested `children` are not used; keep the graph flat"
            )
        if not _ID_RE.match(node.id):
            problems.append(
                f"task {node.id!r}: ids may use letters, digits, '_', '.', '-' and must "
                "start with a letter or digit"
            )
        node_tags = _task_tags(node.type, raw_tags.get(node.id), problems, node.id)
        tags[node.id] = node_tags
        is_merge = "merge" in node_tags
        if is_merge != (node.kind == "merge"):
            problems.append(f"task {node.id!r}: tag `merge` and `kind: merge` must go together")
        if len(node.depends_on) > 1 and not is_merge:
            problems.append(
                f"task {node.id!r} depends on {len(node.depends_on)} tasks; a task that "
                "joins several branches must be tagged `merge`"
            )
        if not _has_task_section(text, node.id):
            problems.append(f"task {node.id!r} has no `### {node.id}` section describing it")

    side = _side_tasks(graph, tags, _raw_field(block, "side"), problems)
    _check_landing(graph, tags, side, problems)

    priorities = default_priorities(graph, tags)
    for node_id, value in raw_priorities.items():
        if value is None or node_id not in priorities:
            continue
        if value not in PRIORITIES:
            problems.append(
                f"task {node_id!r}: `priority` must be one of {', '.join(PRIORITIES)}, "
                f"got {value!r}"
            )
        elif node_id in side and value != "perf":
            problems.append(f"task {node_id!r} is `side`, so its `priority` is perf")
        else:
            priorities[node_id] = value
    for node_id in side:
        priorities[node_id] = "perf"
    spec = PlanSpec(graph=graph, tags=tags, text=text, priorities=priorities, side=side)
    if previous is not None and states is not None:
        for node_id, state in states.items():
            frozen = state in (NodeState.DONE, NodeState.RUNNING)
            if not frozen or node_id not in previous.graph.by_id:
                continue
            label = state.name
            if node_id not in graph.by_id:
                problems.append(f"task {node_id!r} is {label} and cannot be removed")
            elif previous.entry(node_id) != spec.entry(node_id):
                remedy = (
                    "add a new task for follow-up work instead"
                    if state == NodeState.DONE
                    else "list it under `cancel` to change it"
                )
                problems.append(
                    f"task {node_id!r} is {label}; its graph entry and section cannot "
                    f"change; {remedy}"
                )

    if problems:
        raise PlanError(problems)
    return spec
