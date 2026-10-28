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
    MODE_PROMOTE,
    TIER_DESCRIPTION,
    TIER_KERNEL_LIBRARY,
    QueueItem,
)
from ..decisions import DECISION_REMINDER
from .tags import tag_section, task_tags


@dataclass(frozen=True)
class PromotionEntry:
    """One catalog entry's observed variants, callers and prior worker reports."""

    entry: str
    shared: tuple[str, ...]
    exists: bool
    variants: tuple[tuple[str, tuple[str, ...]], ...]
    reports: tuple[str, ...]
    copies: tuple[tuple[str, str], ...]
    importers: tuple[tuple[str, tuple[str, ...]], ...]


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
    promotions: tuple[PromotionEntry, ...]
    reports_dir: str
    environment_kind: str
    devices: int


KERNEL_SOURCES = """\
## Where a kernel comes from

An op in `torch.ops.trtllm` is registered from Python (`tensorrt_llm/_torch/custom_ops/*.py`:
the file name is its source) or from C++ (`cpp/tensorrt_llm/thop/*.cpp`), and grepping the op's
name there finds it. A FlashInfer callable no op wraps is tier 3: the entry's `__call__` calls
`flashinfer.<module>.<name>` itself. What the entry certifies follows the source: a JIT-compiled
kernel (CuTe DSL, Triton, FlashInfer) states the architecture it compiled for and that the JIT
path ran; a compiled-in C++ kernel is in the wheel or not. Say which in the entry's `note`.
"""


def _tier(value: Any) -> int | None:
    """A sourcing rung, when ``value`` is one."""
    try:
        tier = int(value)
    except (TypeError, ValueError):
        return None
    return tier if tier in TIER_DESCRIPTION else None


def _landing_block(ctx: KernelContext) -> str:
    """Where the dispatch's files go when that is a local variant; empty otherwise."""
    item, landing, paths = ctx.item, ctx.landing, ctx.paths
    if paths is None or landing == item.target:
        return ""
    files = "\n".join(f"- `{path}`" for path in paths)
    category, name = item.target.strip("/").split("/")
    shared = " and ".join(f"`{path}`" for path in ctx.shared)
    start = (
        "Write the entry there"
        if item.mode == MODE_ONBOARD
        else (
            "Start the variant as a copy of the entry as it stands, file and test (the "
            "shared catalog's, or the copy another target keeps in its `ops/`), keeping its "
            f"class and its instance name (`{name}`), and add your cells there"
        )
    )
    return (
        "## Where it lands\n\n"
        f"In this target's own `ops/`, as the local variant `{landing}`:\n\n"
        f"{files}\n\n"
        f"{start}. Leave {shared} as they are: other tasks on this "
        f"entry run beside yours, each in a variant of its own. The target imports the "
        f"variant (`from .{landing.replace('/', '.')} import {name}`); before delivery one "
        f"`promote` task merges every variant of `{category}/{name}` into the catalog.\n"
    )


def _promote_block(ctx: KernelContext) -> str:
    """The entries, variants and callers observed on the integration checkout."""
    blocks = ["## What to merge\n"]
    for entry in ctx.promotions:
        lines = [
            f"### `{entry.entry}`\n",
            f"- the shared entry: {', '.join(f'`{path}`' for path in entry.shared)}"
            + ("" if entry.exists else " (not in the catalog yet: you create it)"),
        ]
        lines += [
            f"- local variant `{target}`: {', '.join(f'`{path}`' for path in files)}"
            for target, files in entry.variants
        ]
        if not entry.variants:
            lines.append("- no local variant on the checkout: say so in your report")
        if entry.reports:
            lines.append(
                "- what the kernel writers that landed it said the merge must keep (each report's "
                f"result, 'For the orchestrator'), in `{ctx.reports_dir}`: "
                + ", ".join(f"`{report}`" for report in entry.reports)
            )
        for path, tests in entry.copies:
            lines.append(
                f"- another target's own copy: `{path}`, tested under `{tests}`: it moves "
                "too, and that target imports the catalog"
            )
        for path, variants in entry.importers:
            if variants:
                lines.append(f"- `{path}` imports {', '.join(f'`{v}`' for v in variants)}")
        blocks.append("\n".join(lines) + "\n")
    return "\n".join(blocks)


def _sourcing(ctx: KernelContext) -> str:
    """The dispatch's sourcing lines: the rung its queue row records and what it means.

    Tiers 1-3 wrap something that exists; 4-5 mean an onboarding writes
    the implementation. Saying which changes what the turn is for. A verify
    row seldom names a tier and parses as tier 1 when it does not, so the
    row as written is read, not the parsed item.
    """
    item = ctx.item
    tier, where = _tier(ctx.tier), f"`vocabulary.yaml` queue row {item.n}"
    if tier is None:
        return (
            "Sourcing: no rung is recorded for this entry. Name the one its `__call__` "
            "wraps in your decision file's `tier`.\n"
        )
    line = f"Sourcing: **tier {tier} — {TIER_DESCRIPTION[tier]}** (recorded in {where}).\n"
    if item.mode == MODE_ONBOARD and tier > HIGHEST_REUSING_TIER:
        line += (
            "This entry has no implementation yet — you are writing it, not "
            "wrapping one. Correct-and-slow is acceptable; the tuning phase "
            "replaces it. Say in the contract what you rejected at the cheaper "
            "rungs.\n"
        )
    elif tier == TIER_KERNEL_LIBRARY:
        line += (
            "The entry's `__call__` calls the FlashInfer callable directly — no trtllm op "
            "wraps it. Find it in the FlashInfer the run's interpreter has installed; "
            "certify the callable as the kernel sources below say.\n"
        )
    return line


def _purpose(ctx: KernelContext) -> str:
    """What the plan asks of this dispatch: the graph task's goal, and its queue row's note."""
    item, goal = ctx.item, ctx.goal
    row_note = (item.note or "").strip()
    blocks = []
    if goal:
        tags = ", ".join(task_tags(ctx.task_kind, ctx.tags))
        blocks.append(f"## Your task: `{ctx.task_id}` (tags: {tags})\n\n{goal}\n")
    if row_note and row_note != goal:
        label = "Its queue row's note" if goal else "The plan's note for this dispatch"
        blocks.append(f"{label}: {row_note}\n")
    return "\n".join(blocks)


def _promote_prompt(ctx: KernelContext, assignment: str) -> str:
    """A promote's dispatch: the entries, what each merges, and the skill with the procedure."""
    item, note = ctx.item, ctx.note
    purpose = _purpose(ctx)
    target_rel = ctx.target_rel
    names = ", ".join(f"`{entry}`" for entry in item.entries)
    retry = f"\n{note}\n" if note else ""
    return (
        f"Workspace: {ctx.workspace}\n"
        f"Repository: {ctx.repo}\n"
        f"Architecture: {ctx.arch}\n" + (f"Target: {target_rel}\n" if target_rel else "") + "\n"
        f"Dispatch {item.n}: **promote** this target's local variants of {names} into the "
        f"shared catalog.\n"
        f"{retry}"
        f"{assignment}"
        + (f"\n{purpose}\n" if purpose else "")
        + f"\n{tag_section(task_tags('kernel', ctx.tags))}\n"
        + f"\n{_promote_block(ctx)}\n"
        f"Read `{ctx.vocabulary_path}` for the call sites each entry serves.\n\n"
        + DECISION_REMINDER
    )


def kernel_prompt(ctx: KernelContext) -> str:
    item, note = ctx.item, ctx.note
    surface = f"The surface to certify: {item.surface}\n" if item.surface else ""
    # Directly under the dispatch line, because it changes what the turn
    # is for: finish what the last attempt started, not begin again.
    retry = f"\n{note}\n" if note else ""
    purpose = _purpose(ctx)
    # The local launch protocol hands every role the same devices, and
    # concurrent kernel writers on one device fail with `Executor creation
    # failed due to insufficient GPU memory`: round-robin by dispatch number.
    # Local only: under a scheduler (a spool, Slurm) the job's devices are
    # assigned for it, which an index set here would mask.
    devices = ctx.devices
    assignment = ""
    if devices > 1 and not scheduler_owns_devices(ctx.environment_kind):
        assignment = (
            f"\n**Use `CUDA_VISIBLE_DEVICES={item.n % devices}` for this dispatch**, not the "
            f"devices in the launch protocol above. This node has {devices} devices and your "
            f"batch is running concurrently; every kernel writer taking the same device turns a "
            f"memory error into the symptom.\n"
        )
    if item.mode == MODE_PROMOTE:
        return _promote_prompt(ctx, assignment)
    tier = _sourcing(ctx)
    # An `ops/<category>/<name>` target lives in this target's own `ops/`,
    # and so does a shared entry's local variant.
    target_rel = ctx.target_rel
    landing = _landing_block(ctx)
    return (
        f"Workspace: {ctx.workspace}\n"
        f"Repository: {ctx.repo}\n"
        f"Architecture: {ctx.arch}\n" + (f"Target: {target_rel}\n" if target_rel else "") + "\n"
        f"Dispatch {item.n}: **{item.mode}** the catalog entry `{item.target}`.\n"
        f"{retry}"
        f"{assignment}"
        + (f"\n{purpose}\n" if purpose else "")
        + f"\n{tag_section(task_tags('kernel', ctx.tags))}\n"
        + (f"\n{landing}\n" if landing else "")
        + f"{surface}"
        f"{tier}\n"
        f"{KERNEL_SOURCES}\n"
        f"Read `{ctx.vocabulary_path}` for the computation step this entry serves "
        f"and the values this checkpoint passes it.\n\n" + DECISION_REMINDER
    )
