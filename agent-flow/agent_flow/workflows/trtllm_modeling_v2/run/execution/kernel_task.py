# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kernel task dispatch and its prepared prompt context."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ...agents.prompts.kernel import KernelContext, kernel_prompt
from ...operators.coverage import (
    MODEL_OPS_DIRNAME,
    UNIT_CORE,
    QueueItem,
    entry_paths,
    shared_entry_paths,
    split_variant,
)
from .handoff import DEFAULT_DECISION_UNIT, TaskResult

if TYPE_CHECKING:
    from ..graph import Task


def _run_one_kernel_writer(
    execution, item: QueueItem, note: str, task: Task, card: str = "", context: str = ""
) -> TaskResult:
    """Dispatch the queue item's turn.

    The decision is numbered by ``item.n``; concurrent entries reserve
    separate paths. A retry or a note forces a fresh turn rather than
    adopting an earlier answer from the same attempt. ``context`` is what the
    task depends on: it follows the row's own text, as in a dispatch of any
    other kind.
    """
    prompt = _kernel_writer_prompt(execution, item, note, task=task, card=card)
    return execution._dispatch_task(
        "kernel_writer",
        decision_unit_of(item),
        task.id,
        "\n\n".join(part.strip() for part in (prompt, context) if part.strip()),
        iteration=item.n,
        label=f"{item.mode} {item.target}",
        fresh=bool(note),
    )


def decision_unit_of(item: QueueItem) -> str:
    """What a queue row's decision is filed under: its deployment, or the target itself.

    A row that names no deployment is the target's. Its file is not named for the queue's
    default word for that: a worker read `core-kernel_writer-3.yaml` as "the core task".
    """
    return DEFAULT_DECISION_UNIT if item.unit == UNIT_CORE else item.unit


def _target_rel(execution) -> str:
    """The run's target directory, repo-relative; empty before preflight names it."""
    return str(getattr(getattr(execution.workflow, "_preflight", None), "target_rel", "") or "")


def _landing(execution, item: QueueItem, task: Task | None) -> tuple[str, tuple[str, ...] | None]:
    """Where ``item``'s dispatch writes: the entry it names as a target, and its files.

    An onboard or verify of a shared entry lands its local variant in the
    target's own ``ops/`` (:attr:`QueueItem.variant`), at the site the
    task's ``touches`` declare when they declare a variant of it. It lands
    in place when the task declares the shared module instead, and when no
    target is known. An entry of the target's own ``ops/`` lands
    where it is.
    """
    target_rel = _target_rel(execution)
    repo = execution.workflow.task_data.get("trtllm_repo_path") or None
    if item.variant is None or not target_rel:
        return item.target, entry_paths(item, target_rel, repo)
    touches = list(task.touches) if task is not None else []
    category, name = item.target.strip("/").split("/")
    ops = f"{target_rel.rstrip('/')}/{MODEL_OPS_DIRNAME}/{category}/"
    placed = item
    for path in touches:
        stem, site = split_variant(Path(path).stem)
        if path.startswith(ops) and path.endswith(".py") and site and stem == name:
            placed = QueueItem(n=item.n, mode=item.mode, target=item.target, site=site)
            break
    else:
        if shared_entry_paths(item.target)[0] in touches:
            return item.target, shared_entry_paths(item.target, repo)
    return str(placed.variant), entry_paths(placed, target_rel, repo)


def _kernel_writer_prompt(
    execution, item: QueueItem, note: str = "", task: Task | None = None, card: str = ""
) -> str:
    """Read the task's inputs once and hand only prepared data to the renderer."""
    workflow = execution.workflow
    target_rel = _target_rel(execution)
    landing, paths = _landing(execution, item, task)
    return kernel_prompt(
        KernelContext(
            item=item,
            workspace=str(workflow.workspace),
            repo=str(workflow.task_data["trtllm_repo_path"]),
            arch=str(workflow.task_data["arch"]),
            target_rel=target_rel,
            vocabulary_path=str(workflow.vocabulary_path),
            note=note,
            goal=card or ((task.goal or "").strip() if task else ""),
            task_id=task.id if task else "",
            task_kind=task.kind if task else "kernel",
            landing=landing,
            paths=paths,
            shared=tuple(shared_entry_paths(item.target)) if landing != item.target else (),
        )
    )
