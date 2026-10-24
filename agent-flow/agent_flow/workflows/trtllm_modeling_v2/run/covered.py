# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Before a smith dispatch: the measured geometry, and a certification the tree already holds.

:func:`before_smith` is the one call a smith dispatch makes before its turn
(``SmithStage._run_one_smith``). It waits for a geometry probe still running
(``run/geometry.py``), so the turn reads measured shapes, and then closes a
certification whose cells the target's checkout already certifies, without
a turn (:func:`close_if_covered`).

A certification is a ``verify`` dispatch that states its required cells (a
queue row's ``cells``, or a ```` ```cells ```` block in its task's goal;
``onboard/coverage.py``). It is covered when the entry module and its test
are at the checkout's ``HEAD``, every required cell is a cell of the
entry's ``CELLS``, on the run's architecture in its ``ARCHS``, and the run's
test cache holds a result of every test of that file that holds at ``HEAD``,
none failing and at least one passed (``checks/testcache.head_results``): a
cell listed in ``CELLS`` is a claim, and the test passing is what certifies
it. Then the
dispatch writes the smith's decision itself -- ``LANDED``, with a summary and
a receipt that say which cells of which commit covered it -- and records it
as a smith's would be, so the task ends ``done`` with that receipt and no
turn. Nothing else changes: the task's claimed checks still run after it.

It is never closed when the dispatch states no cells, when anything it would
read cannot be read, or when the task's branch already carries commits of
its own (those are work to integrate, not a certification to skip).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from rich.markup import escape as escape_markup

from agent_flow.console import print_message

from ..checks import testcache
from ..onboard import coverage as coverage_mod
from ..onboard.vocabulary import MODE_VERIFY, QueueItem, entry_paths
from . import geometry
from .progress import collect_decision_file, mark_decision_dispatch

if TYPE_CHECKING:
    from .graph import Task

#: The decision a covered certification is closed with.
COVERED_DECISION = "LANDED"


def _tested(stage: Any, repo: str, head: str, test: str) -> testcache.HeadResults:
    """What the run's test cache holds of ``test`` at ``head``; ``INCOMPLETE`` when it cannot be read."""
    try:
        cache = testcache.open_cache(Path(stage.workspace) / testcache.CACHE_DIRNAME, repo, head)
        return testcache.head_results(cache, head, [test])
    except (testcache.ResultCacheError, OSError, ValueError) as exc:
        return testcache.HeadResults(head, [], [], [], [f"{test} (the cache: {exc})"])


def before_smith(stage: Any, item: QueueItem, task: Task | None = None) -> str | None:
    """What a smith dispatch does first; a result when the dispatch is closed without a turn."""
    geometry.await_probe(stage)
    return close_if_covered(stage, item, task)


def _row(stage: Any, item: QueueItem) -> dict | None:
    raw = getattr(stage, "_raw_vocabulary", None)
    vocabulary = raw() if callable(raw) else {}
    rows = vocabulary.get("queue") if isinstance(vocabulary, dict) else None
    return next(
        (r for r in rows or () if isinstance(r, dict) and str(r.get("n")) == str(item.n)),
        None,
    )


def close_if_covered(stage: Any, item: QueueItem, task: Task | None = None) -> str | None:
    """Close a covered certification as its smith would; ``None`` when it is dispatched.

    ``stage`` is the workflow: its spec, workspace, journal, progress, git
    runner (``_git``), integration and preflight. See the module docstring.
    """
    if item.mode != MODE_VERIFY:
        return None
    text = (task.goal if task is not None else "") or item.note or ""
    required, where = coverage_mod.required_cells(_row(stage, item), text)
    if not required:
        return None
    label = f"{task.id} ({item.target})" if task is not None else f"{item.n} ({item.target})"

    def dispatched(why: str) -> None:
        stage.journal.event("certification dispatched", f"{label}: {why}")

    integration = getattr(stage, "_integration", None)
    if task is not None and integration is not None and integration.changed_paths(task.id):
        dispatched("its branch carries commits of its own")
        return None
    target_rel = str(getattr(getattr(stage, "_preflight", None), "target_rel", "") or "")
    paths = entry_paths(item, target_rel)
    repo = str(stage.task_data.get("trtllm_repo_path") or "")
    arch = str(stage.task_data.get("arch") or "")
    if paths is None or not repo or not arch:
        dispatched("the entry, the checkout or the architecture is not known")
        return None
    entry, test = paths
    head = (stage._git(repo, "rev-parse", "HEAD") or "").strip()
    source = stage._git(repo, "show", f"{head}:{entry}") if head else None
    if not source:
        dispatched(f"`{entry}` is not at the checkout's HEAD")
        return None
    if stage._git(repo, "cat-file", "-e", f"{head}:{test}") is None:
        dispatched(f"`{test}`, the test that drives its cells, is not at the checkout's HEAD")
        return None
    result = coverage_mod.coverage(required, coverage_mod.read_entry_cells(source), arch)
    if not result.covered:
        dispatched(result.reason)
        return None
    tested = _tested(stage, repo, head, test)
    if tested.verdict != "CLEAN":
        dispatched(f"`{test}` at {head[:10]} in the run's test cache: {tested.describe()}")
        return None

    lines = [
        f"{coverage_mod.describe_cell(want)} by cell {cell.index} ({cell.why})"
        for want, cell in result.matches
    ]
    summary = (
        f"Closed without a smith turn: {where} names {len(required)} cell(s), and `{entry}` at "
        f"{head[:10]} already certifies every one on {arch}: its ARCHS names it and its test, "
        f"`{test}`, drives its CELLS. " + "; ".join(lines)
    )
    payload = {
        "decision": COVERED_DECISION,
        "entry": entry,
        "summary": summary,
        "receipt": {
            "result": f"covered at {head[:10]}; no test ran for this task, the entry's own test "
            f"drives the covering cells and its recorded results hold there: {tested.describe()}",
            "covered_by": lines,
        },
        "covered": {
            "tree": head,
            "arch": arch,
            "stated_in": where,
            "cells": [
                {"required": dict(want), "cell": cell.index, "why": cell.why}
                for want, cell in result.matches
            ],
        },
    }
    path = mark_decision_dispatch(stage.workspace, "smith", item.n, unit=item.unit)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=100), encoding="utf-8"
    )
    decision = collect_decision_file(
        stage._progress_ctx, stage.workspace, "smith", item.n, unit=item.unit
    )
    if decision != COVERED_DECISION:
        dispatched(f"its covered decision at `{path}` could not be recorded")
        return None
    stage._smiths_dispatched.add(item.n)
    stage.journal.event(
        "certification closed as covered",
        f"{label}: {len(required)} required cell(s) in `{entry}` at {head[:10]}; `{path}`",
    )
    print_message(
        f"[bold green]✔ {escape_markup(label)}[/bold green] covered at {head[:10]}: "
        f"{len(required)} required cell(s) already in CELLS, no smith turn",
        None,
    )
    return f"covered at {head[:10]}: {'; '.join(lines)}"


__all__ = ["COVERED_DECISION", "before_smith", "close_if_covered"]
