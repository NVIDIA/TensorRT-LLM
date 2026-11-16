# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``trtllm-modeling-v2 resolve``: the outer session's answer to a block.

    resolve --workspace W retry <task> --note "what to do differently" [--task T]
    resolve --workspace W retry run --note "why the final pass should run again" [--task T]
    resolve --workspace W edit-env <task|run> --set environment.env.TMPDIR=/var/tmp --note "..."
    resolve --workspace W split <task> --rows rows.yaml --note "..."
    resolve --workspace W allow-shared <task> --path <repository path> --note "..."
    resolve --workspace W skip <task> --reason "..." [--nice-to-have]
    resolve --workspace W change-requirement <task|run> --set <spec key>=<value> --reason "..."
    resolve --workspace W change-requirement <task|run> --deployment D --benchmark B --anchor V --reason "..."
    resolve --workspace W pool-retry pool --note "..." [--close-unknown SEQ]
    resolve --workspace W pool-adjust pool --set qos=QOS --set time=02:00:00 --note "..."

A verb names what it answers: a blocked task, ``run`` (a final pass that did not hold and
that no task owns) or ``pool``. ``retry`` and ``skip`` also name a task that is running:
its turn is ended, then the verb is applied (``escalation.ENDS_A_TURN``). ``monitor wait``
says which is blocked and gives the file that holds its question; ``monitor status`` lists
everything that waits.

Only one process ever writes the graph. With no controller running this command takes the
workspace lock and applies the verb itself. With one running it leaves the answer beside
the task's receipt (``tasks/<name>.answer.yaml``); the controller applies it on its next
pass and says in that file what became of it, and this command waits for that
(``--wait`` seconds). A controller that leaves first frees the workspace, and the answer
is applied here.

Exit 0: applied, and what it did is printed. 1: not applied, with why. 2: a usage error.
3: left for the controller, which has not applied it within ``--wait``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import yaml

from ..infra.git import CommitPolicy
from . import escalation as escalation_mod
from . import graph as graph_mod
from .records.journal import Journal
from .validation import contract as contract_mod

#: How long the command waits for a running controller's result, unless told otherwise.
DEFAULT_WAIT_S = 120.0
LOCK_FILENAME = ".orchestrator.lock"


def _lock(workspace: Path):
    """The workspace lock, when no controller holds it; ``None`` while one does."""
    import fcntl

    handle = (workspace / LOCK_FILENAME).open("a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    return handle


def _handed_over(workspace: Path, name: str, at: str, timeout_s: float) -> dict[str, Any] | None:
    """What the running controller made of the answer left at ``at``; ``None`` after ``timeout_s``.

    A controller that leaves before it applies the answer (a run that stops with everything
    left blocked is answered at that very moment) frees the workspace: the answer is then
    applied here, as it is when no controller runs.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        found = escalation_mod.wait_settled(workspace, name, at, 0.0)
        if found is not None or time.monotonic() >= deadline:
            return found
        time.sleep(1.0)
        lock = _lock(workspace)
        if lock is not None:
            try:
                apply_offline(workspace)
            finally:
                lock.close()
            return escalation_mod.wait_settled(workspace, name, at, 0.0)


def offline_hooks(workspace: Path, graph: graph_mod.Graph) -> escalation_mod.Hooks:
    """What a command reaches when no controller runs: the workspace's spec, and rows to tasks."""
    from . import orchestrator as orchestrator_mod

    spec_path = workspace / "task.yaml"
    try:
        loaded = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        loaded = None
    task_data: dict[str, Any] = loaded if isinstance(loaded, dict) else {}
    survey = next((t for t in graph.tasks if t.kind == "survey"), None)
    own = graph_mod.own_paths(survey.ref if survey is not None else "")
    try:
        allowed = CommitPolicy.from_task(task_data, workspace=workspace).allowed_paths
    except Exception:  # a spec with no repository still resolves the other verbs
        allowed = ()

    def make_tasks(raw: Any, parent: graph_mod.Task):
        return orchestrator_mod.split_tasks(
            raw,
            parent,
            graph,
            workspace=workspace,
            own_paths=own,
            allowed_paths=allowed,
            repo_root=str(task_data.get("trtllm_repo_path") or ""),
        )

    def respec(keys) -> None:
        # The agents' copy; the next start reads the user's file and puts every recorded
        # change back over it.
        spec_path.write_text(
            yaml.safe_dump(task_data, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        # The contract's file follows: a check run before the next start is recorded with
        # the inputs it is judged by now.
        if (workspace / contract_mod.CONTRACT_FILENAME).is_file():
            try:
                contract_mod.write_contract(workspace, contract_mod.compile_contract(task_data))
            except Exception:  # the next start compiles it from the spec whatever is here
                pass

    def void(keys, why: str) -> list[str]:
        from . import evidence as evidence_mod

        return evidence_mod.void_passes(contract_mod.ContractStatus.load(workspace), keys, why)

    return escalation_mod.Hooks(
        task_data=task_data, make_tasks=make_tasks, respec=respec, void=void
    )


def _graph(workspace: Path) -> graph_mod.Graph:
    path = workspace / graph_mod.GRAPH_FILENAME
    return graph_mod.Graph.load(path) if path.exists() else graph_mod.Graph()


def apply_offline(workspace: Path, answer: Mapping[str, Any] | None = None) -> list[str]:
    """Apply ``answer``, else every answer left and not settled, under the lock the caller holds.

    What each did. An answer given here is applied with no file: nothing has to cross to
    another process. Raises :class:`~.escalation.ResolveError` for one that cannot apply.
    """
    graph = _graph(workspace)
    path = workspace / graph_mod.GRAPH_FILENAME
    hooks = offline_hooks(workspace, graph)
    # The run's journal has every answer, whoever applied it: here, with no controller.
    journal = Journal(path=workspace / "journal.md")

    def log(said: str) -> None:
        journal.event("resolve", said)

    if answer is None:
        return escalation_mod.apply_pending(workspace, graph, hooks, lambda: graph.save(path), log)
    did = escalation_mod.apply_answer(workspace, graph, answer, hooks)
    graph.save(path)
    log(f"[answer] {answer['name']}: {did}")
    return [did]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trtllm-modeling-v2 resolve",
        description=__doc__.split("\n\n")[0],
        epilog="No verb is forbidden and no value is refused: a requirement that is mistaken "
        "may be changed. Every change of the spec is recorded with what changed, from what, "
        "to what and why, and the run's report lists them.",
    )
    parser.add_argument("--workspace", type=Path, required=True, help="the run's workspace")
    parser.add_argument("verb", help=f"one of: {', '.join(escalation_mod.VERBS)}")
    parser.add_argument(
        "name",
        help=f"what the verb answers: a blocked task (for retry and skip, a running one too), "
        f"`{escalation_mod.RUN}` (a final pass that did not hold) or `{escalation_mod.POOL_NAME}`",
    )
    parser.add_argument("--note", help="what you found and what should happen: every verb's reason")
    parser.add_argument("--reason", help="the same as --note (skip, change-requirement)")
    parser.add_argument("--task", help="another task to send back with the one the verb names")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="a spec key (dotted; a list item by its name) and its new value, read as YAML",
    )
    parser.add_argument(
        "--nice-to-have",
        action="store_true",
        help="skip: the task was not required after all, so what depends on it goes on",
    )
    parser.add_argument("--rows", type=Path, help="split: a YAML file of typed rows")
    parser.add_argument("--path", help="allow-shared: the repository path the task may change")
    parser.add_argument("--deployment", help="change-requirement: the anchor's deployment")
    parser.add_argument("--benchmark", help="change-requirement: the anchor's benchmark")
    parser.add_argument("--anchor", help="change-requirement: the anchor's new value")
    parser.add_argument("--unit", help="change-requirement: the anchor's unit")
    parser.add_argument(
        "--close-unknown", type=int, metavar="SEQ", help="pool-retry: a record to close"
    )
    parser.add_argument(
        "--wait",
        type=float,
        default=DEFAULT_WAIT_S,
        help="seconds to wait for a running controller to apply the answer",
    )
    return parser


def _arguments(args: argparse.Namespace) -> dict[str, Any]:
    """The verb's arguments as the answer keeps them: only what was given."""
    out: dict[str, Any] = {}
    note = args.note or args.reason
    if note:
        out["note"] = note.strip()
    if args.task:
        out["task"] = args.task
    if args.nice_to_have:
        out["nice_to_have"] = True
    if args.set:
        out["set"] = escalation_mod.parse_assignments(args.set)
    if args.rows is not None:
        try:
            out["rows"] = yaml.safe_load(args.rows.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise escalation_mod.ResolveError(f"--rows {args.rows} cannot be read: {exc}") from exc
    if args.path:
        out["path"] = args.path.strip()
    if args.anchor is not None:
        out["anchor"] = yaml.safe_load(args.anchor)
        out["deployment"], out["benchmark"] = args.deployment or "", args.benchmark or ""
        if args.unit:
            out["unit"] = args.unit
    if args.close_unknown is not None:
        out["close_unknown"] = args.close_unknown
    return out


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    workspace = args.workspace.resolve()
    said = f"{args.verb} {args.name}"
    try:
        answer = {"name": args.name, "verb": args.verb, "args": _arguments(args)}
        lock = _lock(workspace)
        if lock is not None:
            # No controller runs: this process is the graph's one writer for the moment.
            try:
                (did,) = apply_offline(workspace, answer)
            finally:
                lock.close()
            print(f"{said}: applied: {did}")
            return 0
    except escalation_mod.ResolveError as exc:
        print(f"{said}: NOT applied: {exc}")
        return 1
    # A controller runs: the answer is for the attempt the task is on now.
    task = _graph(workspace).by_id().get(args.name)
    left = escalation_mod.leave_answer(
        workspace, args.name, args.verb, answer["args"], task.attempts if task else 0
    )
    print(f"→ {said}: left for the running controller", file=sys.stderr)
    settled = _handed_over(workspace, args.name, str(left["at"]), args.wait)
    if settled is None:
        print(
            f"{said}: left for the controller, which has not applied it yet. It applies it on "
            f"its next pass: `monitor --workspace {workspace} status` says what still waits"
        )
        return 3
    print(
        f"{said}: {'applied' if settled.get('applied') else 'NOT applied'}: {settled.get('said')}"
    )
    return 0 if settled.get("applied") else 1


if __name__ == "__main__":
    sys.exit(main())
