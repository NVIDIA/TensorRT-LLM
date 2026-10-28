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
"""Rendering for the memory monitor.

Three rules the renderer enforces, because each corresponds to a way these
reports mislead in practice:

* An unmeasurable quantity renders as ``unsupported`` or ``unavailable``,
  never as ``0.00 GiB``.
* ``reserved - allocated`` is labelled "reserved, not handed out", not
  "fragmentation": it also holds blocks awaiting release on other streams.
* A negative residual is printed with its sign and flagged, not clamped.
"""

from .capability import Capability
from .frame import Frame, derived_entries
from .schema import Category, Entry, Placement

__all__ = ["UNAVAILABLE", "UNSUPPORTED", "render"]

_GIB = float(1 << 30)

UNKNOWN = "       unknown"
UNAVAILABLE = "   unavailable"
UNSUPPORTED = "   unsupported"


def _gib(value: int | None) -> str:
    if value is None:
        return UNKNOWN
    return f"{value / _GIB:10.2f} GiB"


def _line(label: str, value: int | None, suffix: str = "") -> str:
    return f"  {label:<26s}{_gib(value)}{suffix}"


def _entry_line(entry: Entry) -> str:
    if entry.size_bytes is None:
        size = UNAVAILABLE
    else:
        size = _gib(entry.size_bytes)
    owner = f" [{entry.owner}]" if entry.owner else ""
    note = f"  ({entry.note})" if entry.note else ""
    return (
        f"  {str(entry.category):<22s}{str(entry.source):<9s}"
        f"{str(entry.basis):<21s}{size}{owner}{note}"
    )


def render(
    frame: Frame,
    capability: Capability,
    *,
    tag: str = "",
    trigger: str = "",
    pid: int | None = None,
    rank: int | None = None,
    device_index: int | None = None,
) -> str:
    """Render a human-readable report."""
    device, torch_r = frame.device, frame.torch
    out: list[str] = []
    add = out.append

    header = "=== TensorRT-LLM GPU memory monitor ==="
    if tag:
        header = f"=== TensorRT-LLM GPU memory monitor: {tag} ==="
    add(header)

    ident = []
    if pid is not None:
        ident.append(f"pid={pid}")
    if rank is not None:
        ident.append(f"rank={rank}")
    if device_index is not None:
        ident.append(f"device={device_index}")
    if trigger:
        ident.append(f"trigger={trigger}")
    if ident:
        add("  " + " ".join(ident))
    add(f"  capability: {capability.describe()}")
    if capability.unsupported_stats:
        add(
            f"  NOTE: backend {capability.allocator_backend!r} leaves "
            f"{len(capability.unsupported_stats)} stats meaningless; "
            "they render as unsupported, not as zero."
        )

    add("")
    add("[device]")
    add(_line("total", device.total))
    add(_line("free", device.free))
    if device.process_used is None:
        add(
            _line(
                "this process",
                device.effective_process_used,
                "   <- UPPER BOUND: NVML unavailable, device-wide figure",
            )
        )
        add(f"  {'other processes':<26s}{UNKNOWN}   <- unknowable without NVML")
    else:
        add(_line("this process", device.process_used))
        add(_line("other processes", device.other_process))

    add("")
    add("[this process]")
    add(_line("torch reserved", torch_r.reserved))
    add(_line("  of which handed out", torch_r.live))
    add(
        _line(
            "  reserved, not handed out",
            torch_r.reserved_unallocated,
            "   <- incl. blocks awaiting release on other streams",
        )
    )
    add(_line("component counters", frame.nontorch_known))
    residual_suffix = ""
    if frame.residual_is_negative:
        residual_suffix = "   <- NEGATIVE, see below"
    add(_line("unexplained", frame.nontorch_residual, residual_suffix))

    torch_entries = [
        e for e in frame.entries if e.placement is not Placement.NON_TORCH and e.counts_toward_total
    ]
    derived = derived_entries(frame)
    add("")
    add("[torch-visible, by category]")
    add(f"  {'category':<22s}{'source':<9s}{'basis':<21s}{'bytes':>14s}")
    for entry in sorted(torch_entries, key=lambda e: -(e.size_bytes or 0)):
        add(_entry_line(entry))
    add(_entry_line(derived[0]))

    component_entries = [
        e for e in frame.entries if e.placement is Placement.NON_TORCH and e.counts_toward_total
    ]
    add("")
    add("[component counters]")
    if not component_entries:
        add("  (none registered)")
    for entry in sorted(component_entries, key=lambda e: -(e.size_bytes or 0)):
        add(_entry_line(entry))
    add(_entry_line(derived[1]))

    excluded = [e for e in frame.entries if not e.counts_toward_total]
    if excluded:
        add("")
        add("[not counted toward any total]")
        for entry in excluded:
            add(_entry_line(entry))

    if frame.unknown_categories:
        add("")
        add("[unmeasured]")
        add(f"  {len(frame.unknown_categories)} provider(s) reported unavailable rather than zero:")
        for category in frame.unknown_categories:
            add(f"    {category}")

    if frame.residual_is_negative:
        add("")
        add(
            "[warning] unexplained memory is negative. Two readings "
            "disagree -- different vantage points, sampling skew, or another"
        )
        add(
            "          process releasing memory mid-measurement. Reported "
            "with its sign rather than clamped, because clamping hides"
        )
        add("          precisely the case worth investigating.")

    return "\n".join(out)


def summarize_by_category(frame: Frame) -> dict[Category, int | None]:
    """Collapse entries to one value per category.

    Categories whose providers reported unavailable map to ``None`` so the
    distinction survives aggregation.
    """
    totals: dict[Category, int | None] = {}
    for entry in frame.entries:
        if not entry.counts_toward_total:
            continue
        if entry.size_bytes is None:
            totals.setdefault(entry.category, None)
            continue
        current = totals.get(entry.category)
        totals[entry.category] = entry.size_bytes + (current or 0)
    for entry in derived_entries(frame):
        totals[entry.category] = entry.size_bytes
    return totals
