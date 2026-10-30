# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The part of the report meant for someone who does not know the allocators.

The detail sections speak in allocator terms -- reserved versus live,
component counters, residuals. A reader chasing an out-of-memory error wants
something else: what the memory is *for*, how big each part is, what grew,
and what to change. This module folds every counted entry into a handful of
groups named for their purpose and renders that view first.

The groups partition the process's memory exactly: torch memory splits into
the tagged categories, the unattributed rest and the cache slack; everything
else is a component counter or the residual. A category that could count
toward a total but has no group would make the rows stop adding up, so the
mapping is required to be total (and a test enforces it).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .frame import Frame
from .schema import Category, Source

__all__ = ["GROUP_OF", "Group", "GroupTotals", "group_totals", "render_summary"]

_GIB = float(1 << 30)

#: Rows smaller than this print as 0.00 GiB and are left out.
_SHOWN = 5 << 20


class Group(Enum):
    """What a block of memory is for, in the reader's terms."""

    KV_CACHE = "KV cache"
    WEIGHTS = "Model weights"
    WORKSPACE = "Runtime buffers"
    TORCH_UNATTRIBUTED = "PyTorch, unattributed"
    TORCH_CACHE = "PyTorch cache (free)"
    CUDA_GRAPHS = "CUDA graphs"
    COMMUNICATION = "Communication buffers"
    OVERHEAD = "CUDA and library overhead"
    OTHER = "Other tagged memory"
    SLEEP_RELEASED = "Released by sleep mode"


_NOTES = {
    Group.KV_CACHE: "allocated up front; free blocks stay resident",
    Group.WEIGHTS: "parameters and buffers of the loaded model(s)",
    Group.WORKSPACE: "buffers owned by named runtime components",
    Group.TORCH_UNATTRIBUTED: "in use by PyTorch, owner not identified; typically activations",
    Group.TORCH_CACHE: "held by the PyTorch allocator but not in use; reusable",
    Group.CUDA_GRAPHS: "held by captured CUDA graphs for replay; not reusable",
    Group.COMMUNICATION: "multi-GPU communication buffers",
    Group.OVERHEAD: "CUDA context, cuBLAS, NCCL internals and other driver memory",
    Group.OTHER: "tagged with a name the report does not group",
    Group.SLEEP_RELEASED: "included in the rows above but returned to the driver",
}

#: Every category that can count toward a total, by the group it belongs to.
#: ``None`` marks a category that is shown for diagnosis but never summed.
GROUP_OF: dict[Category, Group | None] = {
    Category.KV_CACHE_POOL: Group.KV_CACHE,
    Category.KV_CACHE_AUX: Group.KV_CACHE,
    Category.WEIGHTS: Group.WEIGHTS,
    Category.DWDP_PAGES: Group.WEIGHTS,
    Category.SAMPLER: Group.WORKSPACE,
    Category.SPECULATIVE: Group.WORKSPACE,
    Category.MODEL_RUNTIME: Group.WORKSPACE,
    Category.EXECUTOR_RUNTIME: Group.WORKSPACE,
    Category.GUIDED_DECODER: Group.WORKSPACE,
    Category.UNATTRIBUTED_LIVE: Group.TORCH_UNATTRIBUTED,
    Category.COMM_MNNVL: Group.COMMUNICATION,
    Category.COMM_SYMM_MEM: Group.COMMUNICATION,
    Category.COMM_DEEPEP: Group.COMMUNICATION,
    Category.COMM_NVLS: Group.COMMUNICATION,
    Category.COMM_NCCL_WINDOW: Group.COMMUNICATION,
    Category.COMM_NCCL: Group.COMMUNICATION,
    Category.COMM_ULYSSES: Group.COMMUNICATION,
    Category.NATIVE_WORKSPACE: Group.WORKSPACE,
    # Expert weight slots the load balancer moved out of PyTorch.
    Category.EPLB: Group.WEIGHTS,
    Category.NONTORCH_RESIDUAL: Group.OVERHEAD,
    Category.TAGGED_OTHER: Group.OTHER,
    Category.CUDA_GRAPHS: Group.CUDA_GRAPHS,
    Category.CPP_BUFFERS: None,
    Category.SLEEP_RELEASED: Group.SLEEP_RELEASED,
}


@dataclass(frozen=True)
class GroupTotals:
    """Bytes per group, plus what could not be measured."""

    sizes: dict[Group, int | None]
    #: Owners of entries whose size is unknown. Their memory is not missing
    #: from the device; it lands in the overhead row, which therefore names
    #: them.
    unmeasured: tuple[str, ...]
    process_total: int
    process_total_is_upper_bound: bool
    #: What each group is made of, by owner, largest first.
    members: dict[Group, list[tuple[str, int]]] = None


def group_totals(frame: Frame) -> GroupTotals:
    from .frame import derived_entries

    sizes: dict[Group, int | None] = {}
    unmeasured: list[str] = []
    members: dict[Group, dict[str, int]] = {}

    for entry in list(frame.entries) + derived_entries(frame):
        if not entry.counts_toward_total:
            continue
        group = GROUP_OF[entry.category]
        if group is None:
            continue
        if entry.size_bytes is None:
            unmeasured.append(entry.owner or str(entry.category))
            continue
        sizes[group] = (sizes.get(group) or 0) + entry.size_bytes
        if entry.source in (Source.DERIVED, Source.RESIDUAL):
            # These rows are subtractions with no owners of their own; their
            # members come only from an explicit breakdown.
            continue
        name = entry.owner or str(entry.category)
        by_owner = members.setdefault(group, {})
        by_owner[name] = by_owner.get(name, 0) + entry.size_bytes

    pools = frame.torch.pools
    if pools is None:
        sizes[Group.TORCH_CACHE] = frame.torch.reserved_unallocated
    else:
        # Only the default pool's free space can serve the next allocation.
        # A graph pool's is held for replay; a MemPool's serves its owner.
        sizes[Group.TORCH_CACHE] = sum(p.free for p in pools if p.kind == "default")
        for pool in pools:
            kind, label, free = pool.kind, pool.label, pool.free
            if kind == "default" or not free:
                continue
            group = Group.CUDA_GRAPHS if kind == "graph" else Group.OTHER
            name = "free in graph pools" if kind == "graph" else f"{label} pool, free"
            sizes[group] = (sizes.get(group) or 0) + free
            by_owner = members.setdefault(group, {})
            by_owner[name] = by_owner.get(name, 0) + free

    # The overhead row is a residual, so it has no owners of its own; split
    # it instead by when each part appeared. Only with a per-process reading:
    # the startup measurements are per process and would not reconcile with
    # a device-wide figure.
    if not frame.device.process_used_is_upper_bound:
        from .startup import residual_breakdown

        parts = residual_breakdown(frame.nontorch_residual)
        if parts:
            members[Group.OVERHEAD] = dict(parts)

    return GroupTotals(
        sizes=sizes,
        unmeasured=tuple(unmeasured),
        process_total=frame.device.effective_process_used,
        process_total_is_upper_bound=frame.device.process_used_is_upper_bound,
        members={
            g: sorted(owners.items(), key=lambda kv: -abs(kv[1])) for g, owners in members.items()
        },
    )


def _size(value: int | None) -> str:
    return "     unknown" if value is None else f"{value / _GIB:8.2f} GiB"


def _delta(now: int | None, before: int | None) -> str:
    if now is None or before is None:
        return ""
    change = (now - before) / _GIB
    return f"{change:+9.2f} GiB"


def render_summary(
    frame: Frame,
    totals: GroupTotals,
    *,
    baseline: dict[Group, int | None] | None = None,
    headline: str = "",
    advice: dict[Group, str] | None = None,
    suggestion: str = "",
    startup_steps: bool = False,
    unattributed_sites: dict[str, int] | None = None,
) -> list[str]:
    """Lines for the summary section, largest group first.

    ``suggestion`` replaces the advice that would follow from the focus
    group, for when the cause is already known (a startup step that failed).
    """
    from . import startup

    if unattributed_sites:
        # Parts sum to the row: whatever no recorded stack covers -- blocks
        # allocated before recording began -- is listed as such.
        row = totals.sizes.get(Group.TORCH_UNATTRIBUTED) or 0
        parts = dict(unattributed_sites)
        parts["no stack recorded"] = row - sum(parts.values())
        members = dict(totals.members or {})
        members[Group.TORCH_UNATTRIBUTED] = sorted(parts.items(), key=lambda kv: -abs(kv[1]))
        totals = GroupTotals(
            sizes=totals.sizes,
            unmeasured=totals.unmeasured,
            process_total=totals.process_total,
            process_total_is_upper_bound=totals.process_total_is_upper_bound,
            members=members,
        )

    out: list[str] = []
    device = frame.device

    if headline:
        out.append(f"  {headline}")

    bound = (
        f" (upper bound: {frame.device.process_missing_reason})"
        if totals.process_total_is_upper_bound
        else ""
    )
    other = (
        f"other processes {_size(device.other_process).strip()}, "
        if device.other_process is not None
        else ""
    )
    out.append(
        f"  This process uses {_size(totals.process_total).strip()}{bound} of "
        f"{_size(device.total).strip()} ({other}free {_size(device.free).strip()})."
    )
    out.append("")

    has_baseline = baseline is not None
    header = f"  {'':<28s}{'size':>12s}{'share':>8s}"
    if has_baseline:
        header += f"{'since warmup':>15s}"
    out.append(header)

    def sort_key(item):
        return -(item[1] or 0)

    for group, size in sorted(totals.sizes.items(), key=sort_key):
        changed = (
            has_baseline and size is not None and abs(size - (baseline.get(group) or 0)) >= _SHOWN
        )
        if size is not None and abs(size) < _SHOWN and not changed:
            continue
        share = (
            f"{100.0 * size / totals.process_total:7.1f}%"
            if size is not None and totals.process_total > 0
            else "        "
        )
        line = f"  {group.value:<28s}{_size(size):>12s}{share}"
        if has_baseline:
            line += f"{_delta(size, baseline.get(group, 0)):>15s}"
        note = _NOTES[group]
        if group is Group.OVERHEAD and totals.unmeasured:
            note += "; also holds what could not be measured: " + ", ".join(totals.unmeasured)
        if group is Group.OVERHEAD and frame.residual_is_negative:
            note = "NEGATIVE: readings disagree; see details"
        out.append(f"{line}   {note}")
        parts = [(n, b) for n, b in (totals.members or {}).get(group, []) if abs(b) >= _SHOWN]
        # A single owner usually just restates the row, but for the two
        # residual rows the name of the only part is the information.
        if len(parts) > 1 or (parts and group in (Group.TORCH_UNATTRIBUTED, Group.OVERHEAD)):
            listed = ", ".join(f"{n} {b / _GIB:.2f}" for n, b in parts[:5])
            more = f", +{len(parts) - 5} more" if len(parts) > 5 else ""
            out.append(f"  {'':<28s}  of which (GiB): {listed}{more}")

    out.append("")
    plan = startup.render_plan_lines(
        device.device_used, totals.sizes.get(Group.KV_CACHE), totals.sizes.get(Group.TORCH_CACHE)
    )
    if plan:
        out.extend(plan)
        out.append("")
    if startup_steps:
        steps = startup.render_stage_lines()
        if steps:
            out.extend(steps)
            out.append("")
    # Advice only when something went wrong: a healthy startup whose largest
    # part is the KV cache is working as designed, and telling its reader to
    # shrink it would be noise at best.
    out.extend(_conclusion(totals, baseline, (advice or {}) if headline else {}, suggestion))
    return out


def _conclusion(
    totals: GroupTotals,
    baseline: dict[Group, int | None] | None,
    advice: dict[Group, str],
    suggestion: str = "",
) -> list[str]:
    known = {g: s for g, s in totals.sizes.items() if s is not None}
    if not known:
        return ["  Conclusion: nothing could be measured."]

    lines = []
    focus: Group | None = None

    if baseline is not None:
        # The allocator's free cache is left out: warmup can leave it large
        # or small, so its change says little about what the workload did.
        growth = {
            g: s - (baseline.get(g) or 0)
            for g, s in known.items()
            if g is not Group.TORCH_CACHE and baseline.get(g) is not None
        }
        # Growth too small to print is not growth; naming it would point
        # the reader at a row showing +0.00.
        grown = {g: d for g, d in growth.items() if d >= _SHOWN}
        if grown:
            focus = max(grown, key=grown.get)
            lines.append(
                f"  Conclusion: since warmup, {focus.value} grew the most "
                f"({grown[focus] / _GIB:+.2f} GiB)."
            )
        else:
            lines.append("  Conclusion: nothing has grown since warmup.")

    if focus is None:
        focus = max(known, key=known.get)
        share = 100.0 * known[focus] / totals.process_total if totals.process_total else 0.0
        lines.append(f"  Largest: {focus.value} ({share:.1f}% of this process).")

    cache = known.get(Group.TORCH_CACHE, 0)
    if totals.process_total and cache > 0.1 * totals.process_total:
        lines.append(
            f"  Note: PyTorch holds {cache / _GIB:.2f} GiB it is not using; an allocation "
            "can still fail if that space is fragmented."
        )

    if suggestion:
        lines.append(f"  Suggestion: {suggestion}.")
    elif focus in advice:
        lines.append(f"  Suggestion: {advice[focus]}.")
    return lines
