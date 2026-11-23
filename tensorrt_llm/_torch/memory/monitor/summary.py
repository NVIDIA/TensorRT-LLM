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

from . import startup
from .frame import DeviceReading, Frame, derived_entries
from .schema import GIB, GROUP_OF, Category, Group, Source, signed_gib

__all__ = ["GROUP_OF", "Group", "GroupTotals", "group_totals", "render_summary"]

#: Rows smaller than this print as 0.00 GiB and are left out.
_SHOWN = 5 << 20


_NOTES = {
    Group.KV_CACHE: "allocated up front; free blocks stay resident",
    Group.WEIGHTS: "parameters and buffers of the loaded model(s)",
    Group.WORKSPACE: "buffers owned by named runtime components",
    Group.TORCH_UNATTRIBUTED: "in use by PyTorch, owner not identified; typically activations",
    Group.TORCH_CACHE: "held by the PyTorch allocator but not in use; reusable",
    Group.CUDA_GRAPHS: "held by captured CUDA graphs for replay; not reusable",
    Group.COMMUNICATION: "multi-GPU communication buffers",
    Group.OVERHEAD: "CUDA context, cuBLAS, NCCL internals and other driver memory",
    Group.OTHER: "tagged with a name the report does not group, or free in a private pool",
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
            if kind == "default" or not free or pool.released:
                continue
            group = Group.CUDA_GRAPHS if kind == "graph" else Group.OTHER
            name = (
                "free in graph pools"
                if kind == "graph"
                else f"free in pool '{label}', reusable only by that pool"
            )
            sizes[group] = (sizes.get(group) or 0) + free
            by_owner = members.setdefault(group, {})
            by_owner[name] = by_owner.get(name, 0) + free

    # The overhead row is a residual, so it has no owners of its own; split
    # it instead by when each part appeared. Only with a per-process reading:
    # the startup measurements are per process and would not reconcile with
    # a device-wide figure.
    if not frame.device.process_used_is_upper_bound:
        parts = startup.residual_breakdown(frame.nontorch_residual)
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


def _released_parts(frame: Frame) -> dict[str, int]:
    """What sleep mode has released, by the name a reader knows it under."""
    parts: dict[str, int] = {}
    for pool in frame.torch.pools or ():
        if pool.released and pool.reserved:
            tag = pool.sleep_tag or pool.label
            name = startup.step_name(tag)
            if name == tag:
                name = tag.replace("_", " ").capitalize()
            parts[name] = parts.get(name, 0) + pool.reserved
    for entry in frame.entries:
        if entry.released and entry.size_bytes:
            group = GROUP_OF.get(entry.category)
            name = group.value if group else str(entry.category)
            parts[name] = parts.get(name, 0) + entry.size_bytes
    return parts


def _released_line(frame: Frame) -> str:
    """One line for what sleep mode has released; the rows leave it out."""
    parts = sorted(
        ((n, b) for n, b in _released_parts(frame).items() if b >= _SHOWN), key=lambda kv: -kv[1]
    )
    if not parts:
        return ""
    listed = ", ".join(f"{name} {size / GIB:.2f} GiB" for name, size in parts)
    return f"  Released by sleep mode, returns on wakeup (not in the rows above): {listed}."


def _kv_pool_bytes(frame: Frame) -> int | None:
    """Resident bytes of the KV cache pools, or ``None`` if none was measured.

    The startup plan sized only the pools: everything else in the KV cache
    group -- buffers its creation step allocated in PyTorch -- is part of
    the plan's non-KV peak, so counting it here would count it twice.
    """
    sizes = [
        e.size_bytes
        for e in frame.entries
        if e.category is Category.KV_CACHE_POOL
        and e.counts_toward_total
        and e.size_bytes is not None
    ]
    return sum(sizes) if sizes else None


def _baseline_of(baseline: dict[Group, int | None], group: Group) -> int | None:
    """A group's size at warmup: absent means it held nothing, ``None`` unknown."""
    return baseline.get(group, 0)


def _size(value: int | None) -> str:
    return "     unknown" if value is None else f"{value / GIB:8.2f} GiB"


def _delta(now: int | None, before: int | None) -> str:
    if now is None or before is None:
        return ""
    return f"{signed_gib(now - before, 9)} GiB"


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

    def sort_key(item: tuple[Group, int | None]) -> int:
        return -(item[1] or 0)

    for group, size in sorted(totals.sizes.items(), key=sort_key):
        before = _baseline_of(baseline, group) if has_baseline else None
        changed = has_baseline and size is not None and before is not None
        changed = changed and abs(size - before) >= _SHOWN
        if size is not None and abs(size) < _SHOWN and not changed:
            continue
        share = (
            f" {100.0 * size / totals.process_total:6.1f}%"
            if size is not None and totals.process_total > 0
            else "        "
        )
        line = f"  {group.value:<28s}{_size(size):>12s}{share}"
        if has_baseline:
            line += f"{_delta(size, before):>15s}"
        note = _NOTES[group]
        if group is Group.OVERHEAD and totals.unmeasured:
            note += "; also holds what could not be measured: " + ", ".join(totals.unmeasured)
        if group is Group.OVERHEAD and frame.residual_is_negative:
            note = "NEGATIVE: readings disagree; see details"
        out.append(f"{line}   {note}")
        parts = [(n, b) for n, b in (totals.members or {}).get(group, []) if abs(b) >= _SHOWN]
        # A single owner usually just restates the row, but for the two
        # residual rows and the catch-all row the name of the only part is
        # the information.
        named_alone = (Group.TORCH_UNATTRIBUTED, Group.OVERHEAD, Group.OTHER)
        if len(parts) > 1 or (parts and group in named_alone):
            listed = ", ".join(f"{n} {b / GIB:.2f}" for n, b in parts[:5])
            more = f", +{len(parts) - 5} more" if len(parts) > 5 else ""
            out.append(f"  {'':<28s}  of which (GiB): {listed}{more}")

    asleep = _released_line(frame)
    if asleep:
        out.append(asleep)
    out.append("")
    # The startup plan describes the awake process; compared with a sleeping
    # one it would only report the released memory as savings.
    plan = (
        None
        if asleep
        else startup.render_plan_lines(
            device.device_used,
            _kv_pool_bytes(frame),
            totals.sizes.get(Group.TORCH_CACHE),
        )
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
    out.extend(
        _conclusion(
            totals,
            baseline,
            (advice or {}) if headline else {},
            suggestion,
            device,
            sites=unattributed_sites,
            alarm=bool(headline),
        )
    )
    return out


def _conclusion(
    totals: GroupTotals,
    baseline: dict[Group, int | None] | None,
    advice: dict[Group, str],
    suggestion: str = "",
    device: DeviceReading | None = None,
    sites: dict[str, int] | None = None,
    alarm: bool = False,
) -> list[str]:
    known = {g: s for g, s in totals.sizes.items() if s is not None}
    grew = 0
    if not known:
        return ["  Conclusion: nothing could be measured."]

    lines = []
    focus: Group | None = None

    if baseline is not None:
        # The allocator's free cache is left out: warmup can leave it large
        # or small, so its change says little about what the workload did.
        growth = {
            g: s - _baseline_of(baseline, g)
            for g, s in known.items()
            if g is not Group.TORCH_CACHE and _baseline_of(baseline, g) is not None
        }
        # Growth too small to print is not growth; naming it would point
        # the reader at a row showing +0.00.
        grown = {g: d for g, d in growth.items() if d >= _SHOWN}
        if grown:
            focus = max(grown, key=grown.get)
            grew = grown[focus]
            lines.append(
                f"  Conclusion: since warmup, {focus.value} grew the most "
                f"({signed_gib(grown[focus])} GiB)."
            )
        else:
            lines.append("  Conclusion: nothing has grown since warmup.")

    if focus is None:
        focus = max(known, key=known.get)
        share = 100.0 * known[focus] / totals.process_total if totals.process_total else 0.0
        lines.append(f"  Largest: {focus.value} ({share:.1f}% of this process).")

    # When other processes hold most of the GPU, this process's own rows do
    # not explain why memory ran out, and tuning them is the wrong fix.
    other = device.other_process if device is not None else None
    if other is not None and other > totals.process_total and device.free < 0.05 * device.total:
        lines.append(
            f"  Note: other processes on this GPU hold {other / GIB:.2f} GiB, more than this "
            f"process ({totals.process_total / GIB:.2f} GiB), leaving {device.free / GIB:.2f} GiB "
            "free; check what else is running on it."
        )

    cache = known.get(Group.TORCH_CACHE, 0)
    if totals.process_total and cache > 0.1 * totals.process_total:
        lines.append(
            f"  Note: PyTorch holds {cache / GIB:.2f} GiB it is not using; an allocation "
            "can still fail if that space is fragmented."
        )

    # Growth after warmup, which already ran the largest batch, is the most
    # specific evidence a report has; when it is large or memory ran out it
    # outranks the generic advice for the row.
    activation = _profiled_activation()
    explained = _growth_suggestion(
        focus,
        grew,
        advice.get(focus),
        sites,
        activation,
        includes_other_processes=totals.process_total_is_upper_bound,
    )
    unexplained = grew
    if focus is Group.TORCH_UNATTRIBUTED and activation:
        unexplained = grew - activation
    if suggestion:
        lines.append(f"  Suggestion: {suggestion}.")
    elif explained and (alarm or unexplained >= _GROWTH_WORTH_NAMING):
        lines.append(f"  Suggestion: {explained}.")
    elif focus in advice:
        lines.append(f"  Suggestion: {advice[focus]}.")
    return lines


#: Growth since warmup large enough to explain even when nothing failed, so
#: a periodic report can point at a leak before it runs out.
_GROWTH_WORTH_NAMING = GIB


def _profiled_activation() -> int | None:
    """PyTorch memory the profiled forward pass allocated, if startup measured it."""
    plan = startup.memory_plan()
    return plan.activation if plan is not None else None


def _growth_suggestion(
    focus: Group,
    grew: int,
    advice: str | None,
    sites: dict[str, int] | None,
    activation: int | None = None,
    includes_other_processes: bool = False,
) -> str:
    """What to do about a row that grew since warmup, or ``""``.

    The warmup baseline is taken at rest, while a report from inside a
    forward pass also holds that step's activations, untagged. Growth in
    PyTorch memory with no owner is therefore called held memory only beyond
    what the profiled forward pass allocated.
    """
    if not grew:
        return ""
    amount = f"{grew / GIB:.2f} GiB"
    if focus is Group.TORCH_UNATTRIBUTED:
        to_reduce = f"; to reduce activation memory, {advice}" if advice else ""
        if activation and grew - activation < _SHOWN:
            return (
                f"PyTorch memory with no known owner grew {amount} after warmup, within the "
                f"{activation / GIB:.2f} GiB the profiled forward pass allocated, so it is most "
                f"likely activations of the step in flight{to_reduce}"
            )
        if activation:
            beyond = (
                f"{(grew - activation) / GIB:.2f} GiB more than the profiled forward pass "
                f"allocated ({activation / GIB:.2f} GiB)"
            )
        else:
            beyond = "and no forward pass was profiled to tell activations apart"
        if sites:
            site, size = max(sites.items(), key=lambda kv: kv[1])
            if 2 * size > grew:
                where = (
                    f"most of it was allocated by {site} ({size / GIB:.2f} GiB) and is still held"
                )
            else:
                where = (
                    f"the largest site still holding memory is {site} "
                    f"({size / GIB:.2f} of {amount})"
                )
            return (
                f"PyTorch memory with no known owner grew {amount} after warmup, {beyond}; "
                f"{where}{to_reduce}"
            )
        return (
            f"PyTorch memory with no known owner grew {amount} after warmup, {beyond}, so "
            "something may be holding on to memory (a cache, a callback, a leak); rerun with "
            f"TLLM_MEM_MONITOR_STACKS=1 to name the code that allocated it{to_reduce}"
        )
    if focus is Group.OVERHEAD and includes_other_processes:
        return (
            f"the overhead row grew {amount} after warmup, but NVML does not list this process, "
            "so that row also holds other processes' memory and its growth may not be this "
            "process's"
        )
    if focus is Group.OVERHEAD:
        return (
            f"memory outside PyTorch grew {amount} after warmup: a library allocating on first "
            "use, or native code holding memory; the overhead row's parts show when it appeared"
        )
    return ""
