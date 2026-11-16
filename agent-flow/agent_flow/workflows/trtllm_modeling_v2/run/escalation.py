# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Blocks and answers: what the run cannot solve itself, and what the outer session says to it.

A task whose worker says ``blocked``, or that the loop cannot take further (a used budget,
a check nobody can take, a write scope, a skeleton, a tree), is ``blocked`` in the graph.
Its question is its own receipt (``tasks/<task>.<attempt>.receipt.yaml``): what was observed against
what was expected, what was tried and ruled out, and the verb that would answer it. Where
no task owns what is stuck, the run itself is blocked (:data:`RUN`: a final pass that did
not hold) or the pool says an event of its own (:data:`POOL_NAME`). Everything that does
not depend on a block goes on.

The outer session that watches the run (``monitor wait`` returns ``TASK_BLOCKED`` or
``RUN_BLOCKED`` at once) answers with one verb of ``trtllm-modeling-v2 resolve``
(:data:`VERBS`), naming the task, ``run`` or ``pool``. Only the controller writes the
graph: with none running ``resolve`` takes the workspace lock and applies the verb itself;
with one running it leaves one answer file (``tasks/<name>.answer.yaml``), which the
controller applies on its next pass and marks with what became of it. An answer is for one
attempt of its task: it is applied once, and is stale once the task has moved on. No
command is stored, nothing has an id, nothing is replayed.

No verb is forbidden and no value is refused: a requirement that is mistaken may be
changed. What the flow owes is the record: every change of the spec is written to
``spec_changes.jsonl`` with what changed, from what, to what and why, and the run's
report lists them.
"""

from __future__ import annotations

import copy
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from ..infra.files import write_text_atomic
from . import graph as graph_mod
from .records.journal import utc_now

#: The notes every worker reads first (``trtllm-modeling-v2 feedback``).
NOTES_FILENAME = "operator_notes.md"
#: Every change a verb made to the spec: what, from, to, why.
CHANGES_FILENAME = "spec_changes.jsonl"
#: Who blocked a task: its worker's receipt, a check or a used budget, a landing.
WORKER, GATE, INTEGRATION = "worker", "gate", "integration"
#: The two names a verb takes that are no task: the run itself, and the pool.
RUN, POOL_NAME = "run", "pool"
#: Where a task's receipt is, and the answer left for it beside it.
TASKS_DIRNAME, ANSWER_SUFFIX = "tasks", ".answer.yaml"
RETRY, EDIT_ENV, SPLIT, ALLOW_SHARED, SKIP = "retry", "edit-env", "split", "allow-shared", "skip"
CHANGE, POOL_RETRY, POOL_ADJUST = "change-requirement", "pool-retry", "pool-adjust"
#: The verbs a task takes while it runs: its turn is ended, then the verb is applied.
ENDS_A_TURN = (RETRY, SKIP)
VERBS: tuple[str, ...] = (
    RETRY,
    EDIT_ENV,
    SPLIT,
    ALLOW_SHARED,
    SKIP,
    CHANGE,
    POOL_RETRY,
    POOL_ADJUST,
)
#: A change of the environment, and one of what the run is held to.
ENVIRONMENT, REQUIREMENT = "environment", "requirement"
#: The longest text one field of a block keeps.
FIELD_CHARS = 1200


#: How the reason of an attempt begins when the outer session's answer started it.
ANSWERED = "the outer session answered"


def sent_back(task: graph_mod.Task) -> bool:
    """Whether ``task``'s attempt was started by an answer of the outer session.

    Such an attempt is a turn: the note is its reason, whatever the task itself would
    find to do.
    """
    return task.status in ("pending", "running") and task.last.startswith(ANSWERED)


class ResolveError(ValueError):
    """A command that cannot be applied as written; nothing was changed."""


def _clip(value: Any, limit: int = FIELD_CHARS) -> str:
    text = str(value or "").strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


# ------------------------------------------------------- notes, spec changes


def append_note(workspace: str | Path, text: str, *, mark: str = "") -> Path:
    """Append ``text``, with its time, to the notes every worker reads first.

    ``mark`` names what wrote it (a command id): a note with the same mark is written once.
    """
    path = Path(workspace) / NOTES_FILENAME
    heading = f"## {utc_now()}" + (f" ({mark})" if mark else "")
    if mark:
        try:
            if f"({mark})" in path.read_text(encoding="utf-8"):
                return path
        except OSError:
            pass
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as notes:
        notes.write(f"\n{heading}\n\n{text.strip()}\n")
    return path


def spec_changes(workspace: str | Path) -> list[dict[str, Any]]:
    """Every change a verb made to the spec, oldest first."""
    try:
        text = (Path(workspace) / CHANGES_FILENAME).read_text(encoding="utf-8")
    except OSError:
        return []
    found = []
    for line in text.splitlines():
        try:
            data = json.loads(line)
        except ValueError:
            continue
        if isinstance(data, dict):
            found.append(data)
    return found


def _walk(data: Any, key: str, write: bool = True) -> tuple[Any, Any, Path | None]:
    """The holder of the last segment of dotted ``key`` in ``data``, that segment, and a file.

    A list of mappings is entered by the ``name`` of one of them, or by its index.
    Mappings on the way are made where there is none. ``options`` that is a file's
    name is entered as the options that file holds: the holder is then the file's
    mapping and the file is returned, for :func:`set_spec` to write it back, so the
    file goes on saying what runs. Without ``write`` (a trial on a copy) the mapping
    takes the name's place in the copy and no file is named.
    """
    from ..spec.deployments import load_options

    file: Path | None = None
    parts = [p for p in key.split(".") if p]
    if not parts:
        raise ResolveError("a spec key is dotted names, such as `environment.env.TMPDIR`")
    holder = data
    for index, part in enumerate(parts[:-1]):
        if isinstance(holder, list):
            holder = _item(holder, part, ".".join(parts[: index + 1]))
        elif isinstance(holder, dict):
            if part == "options" and isinstance(holder.get(part), str):
                where = ".".join(parts[: index + 1])
                try:
                    held = load_options(holder[part])
                except ValueError as exc:
                    raise ResolveError(f"`{where}` names an options file that {exc}") from exc
                if write:
                    file = Path(holder[part])
                else:
                    holder[part] = held
                data = holder = held
                continue
            if holder.get(part) is None:
                holder[part] = {}
            holder = holder[part]
        else:
            raise ResolveError(f"`{'.'.join(parts[:index])}` of the spec holds no keys")
    last: Any = parts[-1]
    if isinstance(holder, list):
        last = holder.index(_item(holder, last, key))
    elif not isinstance(holder, dict):
        raise ResolveError(f"`{'.'.join(parts[:-1])}` of the spec holds no keys")
    if file is not None:
        return holder, last, _OptionsFile(file, data)
    return holder, last, None


@dataclass(frozen=True)
class _OptionsFile:
    """An options file and the mapping read from it, to be written back after a key is set."""

    path: Path
    options: dict

    def save(self) -> None:
        write_text_atomic(self.path, yaml.safe_dump(self.options, sort_keys=False))


def _item(items: list, part: str, where: str) -> Any:
    for item in items:
        if isinstance(item, Mapping) and str(item.get("name")) == part:
            return item
    if part.isdigit() and int(part) < len(items):
        return items[int(part)]
    names = [str(i.get("name")) for i in items if isinstance(i, Mapping) and i.get("name")]
    raise ResolveError(f"`{where}`: the spec's list has no item named {part!r} (it has {names})")


def spec_class(key: str) -> str:
    """Whether a spec key is the environment's or a requirement: how its change is recorded."""
    parts = key.split(".")
    if parts[0] == "environment" or "env" in parts[1:] or parts[-1] == "timeout":
        return ENVIRONMENT
    return REQUIREMENT


def set_spec(task_data: dict, key: str, value: Any, write: bool = True) -> Any:
    """Set dotted ``key`` of the spec in place; what it held before (``None`` for nothing).

    A key under ``options`` that is a file's name is set in that file (:func:`_walk`).
    """
    holder, last, file = _walk(task_data, key, write)
    before = holder[last] if isinstance(holder, list) else holder.get(last)
    holder[last] = value
    if file is not None:
        file.save()
    return before


def record_change(
    workspace: str | Path,
    task_data: dict,
    key: str,
    value: Any,
    *,
    why: str,
    command: str,
    escalation: str = "",
    kind: str = "",
) -> dict[str, Any]:
    """Change the spec in place and record it once per command and key; the record."""
    held = [
        c for c in spec_changes(workspace) if c.get("command") == command and c.get("key") == key
    ]
    if held:
        set_spec(task_data, key, held[0].get("to"))
        return held[0]
    before = set_spec(task_data, key, value)
    record = {
        "key": key,
        "class": kind or spec_class(key),
        "from": before,
        "to": value,
        "why": why,
        "command": command,
        "escalation": escalation,
        "at": utc_now(),
    }
    path = Path(workspace) / CHANGES_FILENAME
    with path.open("a", encoding="utf-8") as log:
        log.write(json.dumps(record, sort_keys=True, default=str) + "\n")
    return record


def reapply_changes(workspace: str | Path, task_data: dict) -> list[dict[str, Any]]:
    """Put every recorded change back on a spec read anew from the file the user wrote."""
    applied = []
    for change in spec_changes(workspace):
        try:
            set_spec(task_data, str(change.get("key")), change.get("to"))
        except ResolveError:
            continue
        applied.append(change)
    return applied


def parse_assignments(pairs: Sequence[str]) -> dict[str, Any]:
    """``key=value`` pairs as a mapping; a value is read as YAML (``8``, ``true``, ``[a, b]``)."""
    out: dict[str, Any] = {}
    for pair in pairs:
        key, sep, raw = str(pair).partition("=")
        if not sep or not key.strip():
            raise ResolveError(f"{pair!r} is not `key=value`")
        try:
            out[key.strip()] = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise ResolveError(f"the value of {key.strip()!r} is not readable: {exc}") from exc
    return out


# ------------------------------------------------------------ the transaction


@dataclass
class Hooks:
    """What a command reaches outside the graph, lent by whoever applies it."""

    #: The spec as the run holds it: changed in place, so the run sees the change.
    task_data: dict = field(default_factory=dict)
    #: Tasks for the typed rows a split names: ``(rows as written, parent task) -> tasks``.
    make_tasks: Callable[[Any, graph_mod.Task], list[graph_mod.Task]] | None = None
    #: The spec changed under these keys: a live run takes the new values.
    respec: Callable[[Sequence[str]], None] | None = None
    #: Take the standing passes of these checks away, saying why, so each is measured
    #: again: ``(check keys, why) -> the keys voided`` (``evidence.void_passes``).
    void: Callable[[Sequence[str], str], list[str]] | None = None
    #: The pool's own retry: ``(close_unknown or None, note) -> what it said``.
    pool_retry: Callable[[int | None, str], str] | None = None
    #: The pool takes the changed ``environment.slurm`` values: ``() -> what it said``.
    pool_adjust: Callable[[], str] | None = None


def _target(
    graph: graph_mod.Graph, record: Mapping[str, Any], args: Mapping[str, Any]
) -> graph_mod.Task:
    name = str(args.get("task") or record.get("task") or "")
    task = graph.by_id().get(name)
    if task is None:
        raise ResolveError(
            f"`{record.get('id')}` is no task"
            + (f" and the graph has no task {name!r}" if name else "")
            + ": this verb names a task"
        )
    return task


def _reopen(task: graph_mod.Task, why: str) -> None:
    """One more attempt of ``task``, past its budget if need be.

    An attempt that ended in its own worker's question, which the outer session then
    answered, costs the task none of its budget: the budget stops a loop of failures,
    and a question that was answered is none. Nor does an attempt the outer session
    ended while it ran: it did not fail, it was told to do otherwise.
    """
    asked = task.status == "running" or (
        task.status == "blocked" and task.block.get("by") == WORKER
    )
    if task.status != "pending":
        task.attempts += 1
    task.budget = max(task.budget + (1 if asked else 0), task.attempts)
    task.status, task.last, task.block = "pending", why, {}


def _unblock(
    graph: graph_mod.Graph, record: Mapping[str, Any], why: str, also: graph_mod.Task | None = None
) -> list[str]:
    """Reopen the task the answer names, and ``also`` when another is named; their ids."""
    tasks = graph.by_id()
    own = tasks.get(str(record.get("task") or ""))
    reopened = []
    for task in (own, also):
        if task is None or task.id in reopened:
            continue
        _reopen(task, why)
        reopened.append(task.id)
    if own is not None and also is not None and also.id != own.id:
        # The edge is made even where `own` already waits for `also` through another
        # task: a join between them that is done holds nothing (a split's parent stays
        # done when one of its children is sent back), and `own` would start at once.
        if also.id not in own.depends_on and also.id not in graph_mod.blocked_by(graph, own.id):
            own.depends_on.append(also.id)
    return reopened


def _apply_verb(
    workspace: str | Path,
    graph: graph_mod.Graph,
    record: Mapping[str, Any],
    verb: str,
    args: Mapping[str, Any],
    hooks: Hooks,
    note: str,
    why: str,
    cmd_id: str,
) -> str:
    """One verb's change to ``graph``; what it did. Raises before it changes anything, or not at all."""
    esc_id = str(record.get("id"))

    if verb == RETRY:
        other = _target(graph, record, args) if args.get("task") else None
        reopened = _unblock(graph, record, why, other)
        if not reopened and esc_id == RUN:
            # The run itself (a final pass that did not hold): nothing of the target is
            # sent back, and the pass runs again on the target as it is. For a failure
            # that was not the target's: a job that was lost, a defect of the flow that a
            # newer flow has put right.
            append_note(workspace, f"On the run ({esc_id}): {note}", mark=cmd_id)
            return "no task is sent back: the final pass runs again on the target as it is"
        if not reopened:
            raise ResolveError(f"`{esc_id}` names no task to retry")
        append_note(workspace, f"On `{', '.join(reopened)}` ({esc_id}): {note}", mark=cmd_id)
        return f"{', '.join(reopened)} run again with the note"

    if verb == SKIP:
        task = _target(graph, record, args)
        if args.get("nice_to_have") and task.priority == "required":
            # The outer session says it was not owed (a row nobody asked for): what
            # depended on it goes on, as after any task that was only nice to have.
            task.priority = "nice_to_have"
            note = f"{note} (it was required; this answer made it nice to have)"
        task.status, task.last, task.block = "skipped", f"skipped: {note}", {}
        held = graph_mod.blocked_by(graph, task.id)
        said = f"{task.id} is skipped and will be reported not done"
        if held and task.priority == "required":
            # Nothing produced what they need: they wait, and are reported not done with it.
            return (
                f"{said}; {', '.join(held)} depend on it and do not start: skip them too, or "
                f"give them what they need, or the run closes partial with them listed"
            )
        return said + (f"; {', '.join(held)} go on without it" if held else "")

    if verb == ALLOW_SHARED:
        task = _target(graph, record, args)
        prefix = str(args.get("path") or "").strip()
        if not prefix:
            raise ResolveError(
                "`allow-shared` needs --path: the repository path the task may change"
            )
        if prefix not in task.touches:
            task.touches.append(prefix)
        policy = hooks.task_data.get("commit_policy")
        allowed = policy.get("allowed_paths") if isinstance(policy, Mapping) else None
        if isinstance(allowed, list) and not graph_mod.inside(prefix, [str(p) for p in allowed]):
            record_change(
                workspace,
                hooks.task_data,
                "commit_policy.allowed_paths",
                [*allowed, prefix],
                why=note,
                command=cmd_id,
                escalation=esc_id,
                kind=REQUIREMENT,
            )
            if hooks.respec is not None:
                hooks.respec(["commit_policy.allowed_paths"])
        _unblock(graph, record, why, task)
        append_note(workspace, f"`{task.id}` may change `{prefix}` ({esc_id}): {note}", mark=cmd_id)
        return f"{task.id} may change {prefix} and runs again"

    if verb == SPLIT:
        task = _target(graph, record, args)
        if hooks.make_tasks is None:
            raise ResolveError("this run cannot make tasks from rows here")
        made = hooks.make_tasks(args.get("rows"), task)
        # The rows are kept only once the graph they make is admissible.
        children, keep = made if isinstance(made, tuple) else (made, None)
        if not children:
            raise ResolveError("the split names no row")
        trial = graph_mod.Graph(tasks=[*graph.tasks, *children])
        parent = trial.by_id()[task.id]
        before = (parent.kind, list(parent.touches), list(parent.depends_on))
        parent.kind, parent.touches = graph_mod.CHECK_KIND, []
        parent.depends_on = [*parent.depends_on, *[c.id for c in children]]
        problems = graph_mod.validate(trial)
        if problems:
            parent.kind, parent.touches, parent.depends_on = before
            raise ResolveError("the split is not admissible: " + "; ".join(problems))
        if keep is not None:
            keep()
        graph.tasks.extend(children)
        _reopen(task, why)
        append_note(
            workspace,
            f"`{task.id}` is split into {', '.join(c.id for c in children)} ({esc_id}): {note}",
            mark=cmd_id,
        )
        return f"{task.id} is now the join of {', '.join(c.id for c in children)}"

    if verb == CHANGE:
        return _change_requirement(workspace, graph, record, args, hooks, note, why, cmd_id)

    if verb in (EDIT_ENV, POOL_ADJUST):
        changes = dict(args.get("set") or {})
        if not changes:
            raise ResolveError(f"`{verb}` needs what to change: --set key=value")
        if verb == POOL_ADJUST:
            changes = {
                (k if k.startswith("environment.") else f"environment.slurm.{k}"): v
                for k, v in changes.items()
            }
        said = _set_keys(workspace, hooks, changes, note, cmd_id, esc_id)
        if verb == POOL_ADJUST and hooks.pool_adjust is not None:
            said.append(hooks.pool_adjust())
        other = _target(graph, record, args) if args.get("task") else None
        reopened = _unblock(graph, record, why, other)
        return "; ".join(said) + (f"; {', '.join(reopened)} run again" if reopened else "")

    if verb == POOL_RETRY:
        if hooks.pool_retry is None:
            return "no controller keeps a pool here: the next start asks for its holder anew"
        close = args.get("close_unknown")
        return hooks.pool_retry(int(close) if close is not None else None, note)

    raise ResolveError(f"{verb!r} is not a verb; the verbs are {', '.join(VERBS)}")


# ----------------------------------------------------------- what is asked


def describe(task: graph_mod.Task, blocks: Sequence[str] = ()) -> str:
    """One line of a blocked task: as what, on which attempt, what it asks, what it holds.

    Or of one that is done and still asks: its question, which holds nothing.
    """
    block = task.block
    if task.status != "blocked":
        return (
            f"{task.id} ({task.kind}), {task.status}, asks and holds nothing: "
            f"{_clip(block.get('question'), 400)}"
        )
    return (
        f"{task.id} ({task.kind}), attempt {task.attempts}, "
        f"{block.get('by') or 'loop'}/{block.get('kind') or 'blocked'}: {_clip(task.last, 300)}"
        + (f"; it asks: {_clip(block['question'], 400)}" if block.get("question") else "")
        + (f"; it holds {', '.join(blocks)}" if blocks else "")
    )


def describe_run(asked: Mapping[str, str]) -> str:
    """One line of what the run itself asks: a final pass that did not hold, on no task."""
    return f"the run, {asked.get('kind') or 'blocked'}: {_clip(asked.get('observed'), 300)}" + (
        f"; it asks: {_clip(asked['question'], 400)}" if asked.get("question") else ""
    )


def waiting(graph: graph_mod.Graph) -> list[str]:
    """Everything that waits for an answer of the outer session, one line each."""
    lines = [
        describe(task, graph_mod.blocked_by(graph, task.id))
        for task in graph.tasks
        if task.status == "blocked"
    ]
    lines += [describe_run(graph.run_blocked)] if graph.run_blocked else []
    return lines + [describe(t) for t in graph.tasks if t.status != "blocked" and t.block]


# ------------------------------------------------------------------ answers


def answer_path(workspace: str | Path, name: str) -> Path:
    """Where the answer to ``name`` (a task, ``run`` or ``pool``) waits for a running controller."""
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    return Path(workspace) / TASKS_DIRNAME / f"{safe}{ANSWER_SUFFIX}"


def _write_answer(workspace: str | Path, answer: Mapping[str, Any]) -> None:
    path = answer_path(workspace, str(answer["name"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(
        path, yaml.safe_dump(dict(answer), sort_keys=False, allow_unicode=True, width=100)
    )


def leave_answer(
    workspace: str | Path, name: str, verb: str, args: Mapping[str, Any], attempt: int
) -> dict[str, Any]:
    """Leave an answer for the running controller; a later one to the same name replaces it.

    ``attempt`` is the attempt of the task it answers: once the task has moved on, the
    answer is stale and changes nothing.
    """
    answer = {
        "name": name,
        "verb": verb,
        "args": dict(args),
        "attempt": int(attempt),
        "at": utc_now(),
    }
    _write_answer(workspace, answer)
    return answer


def read_answer(workspace: str | Path, name: str) -> dict[str, Any] | None:
    """The answer left for ``name``, settled or not; ``None`` when there is none."""
    try:
        data = yaml.safe_load(answer_path(workspace, name).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    return data if isinstance(data, dict) and data.get("verb") else None


def pending_answers(workspace: str | Path) -> list[dict[str, Any]]:
    """The answers left and not settled yet, oldest first."""
    found = []
    for path in sorted((Path(workspace) / TASKS_DIRNAME).glob(f"*{ANSWER_SUFFIX}")):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            continue
        if isinstance(data, dict) and data.get("verb") and "applied" not in data:
            found.append(data)
    return sorted(found, key=lambda answer: str(answer.get("at") or ""))


def apply_answer(
    workspace: str | Path, graph: graph_mod.Graph, answer: Mapping[str, Any], hooks: Hooks
) -> str:
    """Apply one answer to ``graph`` in place; what it did, in a line.

    Raises :class:`ResolveError` for one that cannot apply, and the graph is then as it
    was: a verb that is none, no reason, a name the graph does not have, a task that is
    not blocked while the run asks nothing (one that runs takes :data:`ENDS_A_TURN`),
    or an answer to an attempt the task has left.
    """
    verb, args = str(answer.get("verb")), dict(answer.get("args") or {})
    name = str(answer.get("name") or "")
    if verb not in VERBS:
        raise ResolveError(f"{verb!r} is not a verb; the verbs are {', '.join(VERBS)}")
    note = str(args.get("note") or "").strip()
    if not note:
        raise ResolveError(f"`{verb}` needs its reason: --note (or --reason)")
    task = graph.by_id().get(name)
    of_pool = verb in (POOL_RETRY, POOL_ADJUST)
    if of_pool != (name == POOL_NAME):
        raise ResolveError(
            f"`{POOL_RETRY}` and `{POOL_ADJUST}` name `{POOL_NAME}`, and it takes no other verb"
        )
    if name == RUN and not graph.run_blocked:
        raise ResolveError("the run asks nothing: no final pass stands blocked")
    if name not in (RUN, POOL_NAME):
        if task is None:
            raise ResolveError(
                f"the graph has no task {name!r}; `{RUN}` and `{POOL_NAME}` are the two "
                "names that are no task"
            )
        asked = int(answer.get("attempt") or task.attempts)
        if task.status == "blocked" and asked != task.attempts:
            raise ResolveError(
                f"`{name}` is blocked on attempt {task.attempts}, and this answers attempt "
                f"{asked}: nothing was changed"
            )
        # A task in its turn takes the two verbs that end it: the session that watches
        # the run decides when an agent cannot, also while the agent is still talking.
        ends = task.status == "running" and verb in ENDS_A_TURN
        if task.status != "blocked" and not task.block and not graph.run_blocked and not ends:
            raise ResolveError(
                f"`{name}` is not blocked (it is {task.status}, attempt {task.attempts}) "
                "and the run asks nothing: nothing was changed"
                + (
                    f". A running task takes `{RETRY}` and `{SKIP}`, which end its turn"
                    if task.status == "running"
                    else ""
                )
            )
    why = f"{ANSWERED} `{name}` with `{verb}`: {note}"
    # The answer's own mark: a change of the spec is recorded once for it, however often a
    # process that stopped half way has it applied again.
    mark = f"{name}@{answer.get('at') or utc_now()}"
    # A task that is done and only asked is not sent back by a requirement or an
    # environment that changed: its work did not wait for the answer.
    asking = task is not None and task.status != "blocked" and bool(task.block)
    stays = asking and verb in (CHANGE, EDIT_ENV)
    asked_of = {"id": name, "task": name if task is not None and not stays else ""}
    did = _apply_verb(workspace, graph, asked_of, verb, args, hooks, note, why, mark)
    if asking:
        task.block = {}
    if name != POOL_NAME:
        # Whatever was asked of the run is answered: the final pass runs again once every
        # task is done or skipped.
        graph.run_blocked = {}
    return did


def apply_pending(
    workspace: str | Path,
    graph: graph_mod.Graph,
    hooks: Hooks,
    save: Callable[[], None],
    log: Callable[[str], None] = lambda said: None,
    hold: Callable[[Mapping[str, Any]], bool] = lambda answer: False,
) -> list[str]:
    """Apply every answer left and not settled yet, oldest first; what each did.

    One that ``hold`` keeps back is left as it is for the next pass: the answer to a
    task whose turn the controller is still ending.

    Each in its own step: the change in one save of the graph, then the answer's own file
    says what became of it, for whoever left it. One that cannot apply is settled too,
    with why: it is not tried on every pass. A process that stopped between the save and
    the settling finds the answer again: its task has moved on, and it is settled as stale.
    """
    said = []
    for answer in pending_answers(workspace):
        if hold(answer):
            continue
        try:
            did, applied = apply_answer(workspace, graph, answer, hooks), True
        except ResolveError as exc:
            did, applied = str(exc), False
        if applied:
            save()
        _write_answer(workspace, {**answer, "applied": applied, "said": did})
        name = answer.get("name")
        line = f"[answer] {name}: {did}" if applied else f"[answer] {name} was not applied: {did}"
        log(line)
        said.append(line)
    return said


def wait_settled(
    workspace: str | Path, name: str, at: str, timeout_s: float, poll_s: float = 0.2
) -> dict[str, Any] | None:
    """The answer to ``name`` left at ``at``, once the controller settled it; ``None`` after ``timeout_s``."""
    deadline = time.monotonic() + timeout_s
    while True:
        answer = read_answer(workspace, name)
        if answer is not None and answer.get("at") == at and "applied" in answer:
            return answer
        if time.monotonic() >= deadline:
            return None
        time.sleep(poll_s)


def _set_keys(
    workspace: str | Path,
    hooks: Hooks,
    changes: Mapping[str, Any],
    note: str,
    cmd_id: str,
    esc_id: str,
    kind: str = "",
) -> list[str]:
    """Set each spec key and record it; what changed, one line each.

    All of them or none: a key that cannot be set refuses the command before any is.
    """
    _check_keys(hooks.task_data, changes)
    said = []
    for key, value in changes.items():
        done = record_change(
            workspace,
            hooks.task_data,
            key,
            value,
            why=note,
            command=cmd_id,
            escalation=esc_id,
            kind=kind,
        )
        said.append(f"{key}: {done['from']!r} -> {done['to']!r} ({done['class']})")
    if changes and hooks.respec is not None:
        hooks.respec(list(changes))
    return said


def _contract_inputs(task_data: Mapping[str, Any]) -> dict[str, str]:
    """Each contract check's inputs for the spec as it is, by check key; empty when none compiles."""
    from .validation import contract as contract_mod

    try:
        contract = contract_mod.compile_contract(copy.deepcopy(dict(task_data)))
    except Exception:  # a spec no contract compiles from names no check to judge again
        return {}
    return {check.key: str(check.inputs) for check in contract.checks}


def _check_keys(task_data: Mapping[str, Any], changes: Mapping[str, Any]) -> None:
    """Raise :class:`ResolveError` unless every key of ``changes`` can be set: all or none."""
    trial = copy.deepcopy(dict(task_data))
    for key, value in changes.items():
        set_spec(trial, str(key), value, write=False)


def _change_requirement(
    workspace: str | Path,
    graph: graph_mod.Graph,
    record: Mapping[str, Any],
    args: Mapping[str, Any],
    hooks: Hooks,
    note: str,
    why: str,
    cmd_id: str,
) -> str:
    """Change what the run is held to, record it, and have what it judged judged again.

    An anchor is changed where the run froze it (``checks/anchors.py:change``: the run
    froze its anchors at its first start, so a spec key alone moves none). Any other
    requirement is a spec key; a set of keys is checked whole before one is changed. Then
    every check the change touches is judged again: the escalation's own task, a task
    named with ``--task``, the session of each deployment the change names (its anchor, or
    a key under ``deployments.<name>``), and every check task one of whose checks is
    judged by other inputs after the change. A pass judged by the old requirement no longer
    counts (the ledger gets a line that says so), and its task is queued for a new load.
    """
    esc_id = str(record.get("id"))
    changes = dict(args.get("set") or {})
    anchored = args.get("anchor") is not None
    if not changes and not anchored:
        raise ResolveError(
            "`change-requirement` needs what to change: --set key=value, or "
            "--deployment D --benchmark B --anchor V"
        )
    said = []
    _check_keys(hooks.task_data, changes)
    # What each check is judged by before the change: a check whose inputs the change
    # moves is judged again, whichever task the escalation was on.
    before = _contract_inputs(hooks.task_data)
    if anchored:
        named_anchor = (str(args.get("deployment") or ""), str(args.get("benchmark") or ""))
        said.append(
            set_anchor(workspace, *named_anchor, args.get("anchor"), args.get("unit"), note)
        )
    said += _set_keys(workspace, hooks, changes, note, cmd_id, esc_id, kind=REQUIREMENT)
    after = _contract_inputs(hooks.task_data)
    moved = {key for key in {*before, *after} if before.get(key) != after.get(key)}
    named = {str(args.get("deployment") or "")} if anchored else set()
    named |= {
        key.split(".")[1] for key in changes if key.startswith("deployments.") and "." in key[12:]
    }
    if any(key.split(".")[-1] == "anchor" for key in changes):
        said.append(
            "note: a run's anchors are frozen at its first start, and a frozen one is "
            "changed with --deployment, --benchmark and --anchor"
        )
    tasks = graph.by_id()
    sessions = {f"deployment:{name}" for name in named if name}
    targets = [
        tasks.get(str(record.get("task") or "")),
        tasks.get(str(args.get("task") or "")),
        # The session of a deployment the change names, and every check task one of whose
        # checks is judged by other inputs now: completed or not, it is measured again.
        *(
            t
            for t in graph.tasks
            if t.kind == graph_mod.CHECK_KIND and (t.check in sessions or moved & set(t.satisfies))
        ),
    ]
    queued = []
    for task in targets:
        if task is None or task.id in queued:
            continue
        queued.append(task.id)
        keys = list(task.satisfies)
        voided = hooks.void(keys, f"{esc_id}: {note}") if hooks.void is not None and keys else []
        _reopen(task, why)
        if keys:
            said.append(
                f"`{task.id}` is queued for a new load"
                + (
                    f", and {len(voided)} pass(es) by the old requirement no longer count"
                    if voided
                    else ""
                )
            )
        else:
            said.append(f"`{task.id}` runs again")
    append_note(
        workspace,
        f"A requirement changed ({esc_id}): {'; '.join(said)}. Why: {note}",
        mark=cmd_id,
    )
    return "; ".join(said)


def set_anchor(
    workspace: str | Path, deployment: str, benchmark: str, value: Any, unit: Any, reason: str
) -> str:
    """Change one frozen anchor through the session step's own command; what it said.

    ``checks.anchors.change`` writes the new anchor and appends what changed, from what,
    to what and why, to ``anchor_changes.jsonl``. ``value`` is the score on the
    benchmark's own scale; ``none`` takes the anchor away (the benchmark is then blocked
    before its load, and said).
    """
    from ..checks import anchors as anchors_mod

    if not deployment or not benchmark:
        raise ResolveError("an anchor is named by --deployment and --benchmark")
    score = None if value is None or str(value).strip().lower() == "none" else value
    # Once per command: one whose change was made and whose graph save was lost is applied
    # again, and finds the change it made.
    last = next(
        (
            c
            for c in reversed(anchors_mod.changes(workspace))
            if c.get("anchor") == f"{deployment}:{benchmark}"
        ),
        None,
    )
    stands = (last or {}).get("to") if isinstance((last or {}).get("to"), Mapping) else None
    if last is not None and last.get("reason") == reason:
        same = (stands or {}).get("value") if stands else None
        if (score is None and stands is None) or (
            score is not None and same is not None and float(same) == float(score)
        ):
            return (
                f"anchor {deployment}:{benchmark}: already changed for this reason "
                f"(recorded in {anchors_mod.CHANGES_FILENAME})"
            )
    try:
        before, after = anchors_mod.change(
            workspace,
            deployment,
            benchmark,
            reason=reason,
            value=None if score is None else float(score),
            unit=str(unit) if unit else None,
        )
    except (ValueError, TypeError, OSError) as exc:
        raise ResolveError(f"the anchor was not changed: {exc}") from exc
    was = "nothing frozen" if before is None else f"{before.kind} {before.value}"
    return (
        f"anchor {after.key}: {was} -> {after.kind} {after.value} "
        f"(recorded in {anchors_mod.CHANGES_FILENAME})"
    )


__all__ = [
    "ALLOW_SHARED",
    "ANSWER_SUFFIX",
    "CHANGE",
    "CHANGES_FILENAME",
    "EDIT_ENV",
    "GATE",
    "Hooks",
    "INTEGRATION",
    "NOTES_FILENAME",
    "POOL_ADJUST",
    "POOL_NAME",
    "POOL_RETRY",
    "RETRY",
    "RUN",
    "ResolveError",
    "SKIP",
    "SPLIT",
    "TASKS_DIRNAME",
    "VERBS",
    "WORKER",
    "answer_path",
    "append_note",
    "describe",
    "describe_run",
    "apply_answer",
    "apply_pending",
    "leave_answer",
    "parse_assignments",
    "pending_answers",
    "read_answer",
    "reapply_changes",
    "record_change",
    "set_anchor",
    "spec_changes",
    "wait_settled",
    "waiting",
]
