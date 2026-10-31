# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kernel task dispatch, prepared prompt context and failed-receipt integration."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from ...agents.prompts.kernel import KernelContext, PromotionEntry, kernel_prompt
from ...operators import promotion as promotion_mod
from ...operators.coverage import (
    MODE_PROMOTE,
    MODEL_OPS_DIRNAME,
    QueueItem,
    entry_paths,
    is_entry_file,
    model_ops_tests_dir,
    shared_entry_paths,
    split_variant,
)
from .handoff import TaskResult, uncertified_report

if TYPE_CHECKING:
    from ..graph import Task

FAILED_RECEIPT = "FAILED_RECEIPT"


def _run_one_kernel_writer(execution, item: QueueItem, note: str, task: Task) -> TaskResult:
    """Dispatch the queue item's turn, then safely integrate any failed receipt.

    The decision is numbered by ``item.n``; concurrent entries reserve
    separate paths. A retry or rebase note forces a fresh turn rather than
    adopting an earlier answer from the same attempt.
    """
    decision = execution._dispatch_task(
        "kernel_writer",
        item.unit,
        task.id,
        _kernel_writer_prompt(execution, item, note, task=task),
        iteration=item.n,
        label=f"{item.mode} {item.target}",
        fresh=bool(note),
    )
    decision.annotate(_land_failed_receipt(execution, item, task, decision.decision))
    if decision.decision == "LANDED":
        decision.annotate(uncertified_report(decision.data.get("uncertified")))
    return decision


def _land_failed_receipt(execution, item: QueueItem, task: Task, decision: str) -> str:
    """Integrate a FAILED_RECEIPT's commits when they change nothing but the entry's files.

    The task's own integration leaves a failed receipt, but what it
    committed (the input `is_valid` now refuses, the guard test that proves
    it, the cells that hold) is knowledge every caller of the entry needs.
    A branch that changes any file outside the entry's (:func:`is_entry_file`,
    its local variant's or the shared entry's) stays as it is, and the
    task's next attempt starts from it. What became of the commits reaches
    the planner with the task's event (:func:`_run_one_kernel_writer`).
    A promote's commits are its task's to integrate, whole or not at all.
    """
    integration = getattr(execution, "_integration", None)
    if decision != FAILED_RECEIPT or integration is None or item.mode == MODE_PROMOTE:
        return ""
    paths = integration.changed_paths(task.id)
    if not paths:
        return ""
    branch = integration.branch(task.id)
    landing, _ = _landing(execution, item, task)
    outside = [
        path
        for path in paths
        if not (is_entry_file(path, item.target) or is_entry_file(path, landing))
    ]
    if outside:
        landed = (
            f"its commits stay on {branch} for the task's next attempt: they change more "
            f"than the entry's own files ({', '.join(outside)})"
        )
    else:
        outcome = integration.integrate(task.id)
        if outcome.merged:
            integration.close(task.id)
        landed = (
            f"its {outcome.commits} commit(s), the entry's own files only, are integrated: "
            f"what it measured, refuses and tests is on the target"
            if outcome.merged
            else f"its commits stay on {branch}: {outcome.detail}"
        )
    execution.workflow.journal.event("integration", f"{task.id}: {FAILED_RECEIPT}, {landed}")
    return landed


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
    where it is; a promote spans several entries and has no one place.
    """
    if item.mode == MODE_PROMOTE:
        return item.target, None
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


def _raw_vocabulary(execution) -> dict:
    """``vocabulary.yaml`` as written, unvalidated; empty when it cannot be read."""
    try:
        data = yaml.safe_load(execution.workflow.vocabulary_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    return data if isinstance(data, dict) else {}


def _kernel_writer_prompt(
    execution, item: QueueItem, note: str = "", task: Task | None = None
) -> str:
    """Read the task's inputs once and hand only prepared data to the renderer."""
    workflow = execution.workflow
    target_rel = _target_rel(execution)
    repo = Path(str(workflow.task_data.get("trtllm_repo_path") or ""))
    landing, paths = _landing(execution, item, task)
    rows = _raw_vocabulary(execution).get("queue") or ()
    row = next((r for r in rows if isinstance(r, dict) and str(r.get("n")) == str(item.n)), {})
    promotions = []
    if item.mode == MODE_PROMOTE:
        variants = promotion_mod.local_variants(repo, target_rel)
        importers = promotion_mod.variant_imports(repo, target_rel)
        models = repo / promotion_mod.MODELING_V2_PACKAGE / "models"
        for entry in item.entries:
            shared = shared_entry_paths(entry, repo)
            category, name = entry.split("/")
            reports = sorted(
                (
                    path.name
                    for mode in ("onboard", "verify")
                    for path in workflow.journal.returned_reports(
                        "kernel_writer", f"{mode} {entry}"
                    )
                ),
                key=lambda report: int(report.split("-", 1)[0]),
            )
            copies = []
            for copy in sorted(models.glob(f"*/*/{MODEL_OPS_DIRNAME}/{category}/{name}.py")):
                other = copy.parents[2].relative_to(repo).as_posix()
                if other != target_rel:
                    copies.append(
                        (
                            copy.relative_to(repo).as_posix(),
                            f"{model_ops_tests_dir(other)}/{category}/",
                        )
                    )
            promotions.append(
                PromotionEntry(
                    entry=entry,
                    shared=tuple(shared),
                    exists=(repo / shared[0]).is_file(),
                    variants=tuple((v.target, v.files) for v in variants if v.entry == entry),
                    reports=tuple(reports),
                    copies=tuple(copies),
                    importers=tuple(
                        (path, tuple(v for v in found if promotion_mod.variant_entry(v) == entry))
                        for path, found in importers.items()
                    ),
                )
            )
    environment = workflow.task_data.get("environment") or {}
    return kernel_prompt(
        KernelContext(
            item=item,
            workspace=str(workflow.workspace),
            repo=str(workflow.task_data["trtllm_repo_path"]),
            arch=str(workflow.task_data["arch"]),
            target_rel=target_rel,
            vocabulary_path=str(workflow.vocabulary_path),
            note=note,
            goal=(task.goal or "").strip() if task else "",
            task_id=task.id if task else "",
            task_kind=task.kind if task else "kernel",
            tags=tuple(task.tags) if task else (),
            tier=row.get("tier"),
            landing=landing,
            paths=paths,
            shared=tuple(shared_entry_paths(item.target))
            if item.mode != MODE_PROMOTE and landing != item.target
            else (),
            promotions=tuple(promotions),
            reports_dir=str(workflow.journal.reports_dir)
            if any(p.reports for p in promotions)
            else "",
            environment_kind=str(environment.get("kind", "local")),
            devices=int(environment.get("gpus_per_node") or 1),
        )
    )
