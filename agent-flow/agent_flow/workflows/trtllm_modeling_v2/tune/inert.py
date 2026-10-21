# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Members a candidate provably does not change: a loss there is the host's, not the candidate's.

A member (perf config) is *inert* for the candidate in one of two ways:

- By the marker: the change's ``path_marker`` printed where its path runs (the
  server log of some measured member), this member's logs lack it, and the engine
  it built reads as the control's (``MV2FLOW_ENGINE`` records, recurrent-state and
  KV-cache pools), or, with no such readings, the decision's ``changed_members``
  leave it out.
- By the members the change runs at: the measurement holds none of the
  ``changed_members``, so the marker has nowhere to print, and this member's
  engine reads as the control's in every run. The list alone is no reading.

A member the decision names is never inert. When the measurement holds one, only a
printed marker shows where the path ran.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Collection, Mapping, Sequence

from .records import Measurement

#: What ``bench.py`` writes per measured perf config: the server's whole output.
SERVER_LOG = "server.log"


def server_log(m: Measurement) -> str:
    """The server output under one perf config's label dir; ``""`` when it cannot be read."""
    try:
        return (Path(m.path) / SERVER_LOG).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def engine_reading(text: str) -> dict[str, Any]:
    """What a server log says the engine built: its records' configuration and pools, its pool lines.

    Free device memory is left out: it differs between two runs of one engine.
    """
    from ..checks.gates import parse_engine_records, parse_kv_allocations, parse_recurrent_slots

    reading: dict[str, Any] = {}
    records = [r for r in parse_engine_records(text) if isinstance(r, Mapping)]
    if records:
        reading["engine"] = {"config": records[-1].get("config"), "pools": records[-1].get("pools")}
    slots = parse_recurrent_slots(text)
    if slots:
        reading["recurrent_slots"] = sorted(slots)
    kv = parse_kv_allocations(text)
    if kv:
        reading["kv_tokens"] = sorted(kv)
    return reading


def _describe(reading: Mapping[str, Any]) -> str:
    parts = []
    if "recurrent_slots" in reading:
        parts.append(f"recurrent slots {reading['recurrent_slots']}")
    if "kv_tokens" in reading:
        parts.append(f"KV-cache tokens {reading['kv_tokens']}")
    if "engine" in reading:
        parts.append("its engine record")
    return ", ".join(parts)


def inert_members(
    candidates: Sequence[Measurement],
    controls: Sequence[Measurement],
    *,
    changed: Collection[str] = (),
    marker: str = "",
) -> dict[str, str]:
    """``{member: why}`` for every member where the candidate provably serves what the control does.

    ``candidates`` and ``controls`` are measurements of the candidate's tree and of the
    control's on one host (each over the same perf configs); ``changed`` and ``marker``
    are the decision's ``changed_members`` and ``path_marker``.
    """
    named = set(changed)
    if not candidates or not candidates[0].parts or not (marker or named):
        return {}
    names = list(candidates[0].parts)
    logs = {
        name: [server_log(m.parts[name]) for m in candidates if name in m.parts] for name in names
    }
    marked = [name for name in names if marker and any(marker in text for text in logs[name])]
    # Without a marker that printed, the decision's list speaks only for a measurement that
    # holds none of the members it names; one that holds them would have shown the marker.
    declared = not marked and bool(named) and not named & set(names)
    if not marked and not declared:
        return {}
    out: dict[str, str] = {}
    for name in names:
        if name in marked or name in named or not logs[name]:
            continue
        theirs = [server_log(m.parts[name]) for m in controls if name in m.parts]
        if not theirs:
            continue
        readings = [engine_reading(text) for text in [*logs[name], *theirs]]
        same = any(readings) and all(r == readings[0] for r in readings)
        if declared:
            if same:
                absent = f"; `{marker}` printed nowhere" if marker else ""
                out[name] = (
                    f"not among the members the change runs at ({', '.join(sorted(named))}), none "
                    f"of which this measurement holds{absent}; {_describe(readings[0])} as the "
                    "control's"
                )
            continue
        absent = f"`{marker}` absent (printed at {', '.join(marked)})"
        if same:
            out[name] = f"{absent}; {_describe(readings[0])} as the control's"
        elif named and not any(readings):
            out[name] = (
                f"{absent}; not among the members the change runs at ({', '.join(sorted(named))})"
            )
    return out


__all__ = ["SERVER_LOG", "engine_reading", "inert_members", "server_log"]
