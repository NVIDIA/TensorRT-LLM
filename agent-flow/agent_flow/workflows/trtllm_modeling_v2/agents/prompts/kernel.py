# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render a kernel task from the context prepared by its run dispatcher."""

from __future__ import annotations

from dataclasses import dataclass

from ...operators.coverage import MODE_ONBOARD, QueueItem
from ..decisions import DECISION_REMINDER
from ._render import render_template
from .tags import tag_section, task_tags


@dataclass(frozen=True)
class KernelContext:
    """Prepared task data; rendering never reads a checkout or mutates a run."""

    item: QueueItem
    workspace: str
    repo: str
    arch: str
    target_rel: str
    vocabulary_path: str
    note: str
    goal: str
    task_id: str
    task_kind: str
    #: Read from the row by the caller and not used: a row states no sourcing rung any more.
    landing: str
    paths: tuple[str, ...] | None
    shared: tuple[str, ...]


def kernel_prompt(ctx: KernelContext) -> str:
    """Render the dispatch. The spool's watcher assigns each job its device, on one host too."""
    local_variant = ctx.paths is not None and ctx.landing != ctx.item.target
    row_note = (ctx.item.note or "").strip()
    return render_template(
        "kernel.md.j2",
        ctx=ctx,
        row_note=row_note if row_note != ctx.goal else "",
        tags=tag_section(task_tags("kernel")),
        local_variant=local_variant,
        onboard=ctx.item.mode == MODE_ONBOARD,
        entry_name=ctx.item.target.strip("/").split("/")[-1],
        variant_module=ctx.landing.replace("/", "."),
        decision_reminder=DECISION_REMINDER,
    )


__all__ = ["KernelContext", "kernel_prompt"]
