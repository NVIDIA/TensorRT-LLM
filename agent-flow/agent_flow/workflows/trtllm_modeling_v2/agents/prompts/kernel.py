# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render a kernel task from the context prepared by its run dispatcher."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...infra.compute.environment import scheduler_owns_devices
from ...operators.coverage import (
    HIGHEST_REUSING_TIER,
    MODE_ONBOARD,
    TIER_DESCRIPTION,
    TIER_KERNEL_LIBRARY,
    QueueItem,
)
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
    tags: tuple[str, ...]
    tier: Any
    landing: str
    paths: tuple[str, ...] | None
    shared: tuple[str, ...]
    environment_kind: str
    devices: int


def _tier(value: Any) -> int | None:
    """A sourcing rung, when ``value`` is one."""
    try:
        tier = int(value)
    except (TypeError, ValueError):
        return None
    return tier if tier in TIER_DESCRIPTION else None


def kernel_prompt(ctx: KernelContext) -> str:
    """Render the dispatch, assigning local devices only when no scheduler owns them."""
    # Concurrent local writers otherwise take the same device. A scheduler
    # assigns devices itself, which a local index would mask.
    device = (
        ctx.item.n % ctx.devices
        if ctx.devices > 1 and not scheduler_owns_devices(ctx.environment_kind)
        else None
    )
    tier = _tier(ctx.tier)
    local_variant = ctx.paths is not None and ctx.landing != ctx.item.target
    row_note = (ctx.item.note or "").strip()
    return render_template(
        "kernel.md.j2",
        ctx=ctx,
        device=device,
        purpose_tags=", ".join(task_tags(ctx.task_kind, ctx.tags)),
        row_note=row_note if row_note != ctx.goal else "",
        tags=tag_section(task_tags("kernel", ctx.tags)),
        local_variant=local_variant,
        onboard=ctx.item.mode == MODE_ONBOARD,
        entry_name=ctx.item.target.strip("/").split("/")[-1],
        variant_module=ctx.landing.replace("/", "."),
        tier=tier,
        tier_description=TIER_DESCRIPTION.get(tier, ""),
        writes_implementation=(
            ctx.item.mode == MODE_ONBOARD and tier is not None and tier > HIGHEST_REUSING_TIER
        ),
        kernel_library=tier == TIER_KERNEL_LIBRARY,
        decision_reminder=DECISION_REMINDER,
    )
