# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime facts: what the engines of passed gates ran, per unit and member.

Every gate that builds an engine attests it (``GateResult.engine``): the
CUDA-graph batches it captured and whether it pads to them, the recurrent-state
slots each rank serves from, its KV-cache pool in tokens, the device memory it
left free, its speculation depth, its widths and its batch and token limits.
Those are kept here, in ``runtime_facts.yaml``.

:func:`record` runs each time a gate passes. The gate's engines become the
latest of that gate (and engine leg) for its unit and member, and the file is
rewritten whole:

- ``members``: per ``<unit>@<member>``, its engines, and the facts taken
  across those measured on the target as it is when the file is written. A
  member with none there keeps the facts of its latest engines, marked
  ``stale``.
- ``envelope``: across every member, the most slots, the largest pool and
  limits, every batch captured, every speculation depth, the least memory
  left free, and where each was measured.
- ``changes``: each time a gate's own numbers moved since its previous pass
  (the capture list, the padding, the slots, the speculation, the widths, the
  limits), newest last. Free memory and the KV pool follow the device's other
  tenants and are not changes.

:func:`render` is the block a prompt carries: the path, the envelope, one
line per member, and the latest changes.
"""

from __future__ import annotations

import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from ..checks.gates import format_batches

FACTS_FILENAME = "runtime_facts.yaml"
#: How many changes the file keeps, newest last.
MAX_CHANGES = 30
#: How many member lines a prompt's block shows; the rest are in the file.
MEMBER_LINES = 12
#: The heading a prompt carries the facts under.
HEADING = "## Runtime facts (measured)"

#: An engine's numbers whose movement between two passes of one gate is a change.
CHANGE_FIELDS: tuple[str, ...] = (
    "cuda_graph_batches",
    "cuda_graph_padded",
    "recurrent_slots",
    "speculative",
    "tp",
    "ep",
    "attention_dp",
    "max_batch_size",
    "max_num_tokens",
)
_LIMITS: tuple[str, ...] = ("max_batch_size", "max_num_tokens")

#: One writer at a time: several checks pass at once, each on a thread of its own.
_LOCK = threading.Lock()


class _Dumper(yaml.SafeDumper):
    """Lists of numbers on one line: a capture list is tens of batches."""


def _represent_list(dumper: yaml.SafeDumper, data: list) -> yaml.Node:
    flow = bool(data) and all(isinstance(v, (int, float)) for v in data)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=flow)


_Dumper.add_representer(list, _represent_list)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def facts_path(workspace: Path) -> Path:
    return Path(workspace) / FACTS_FILENAME


def _int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def engine_numbers(record: Mapping[str, Any]) -> dict[str, Any]:
    """The numbers one engine record (``EngineAttestation.record``) carries; empty when none.

    The batches are those the engine captured where its worker was read, and
    those it was configured with otherwise. A field the record does not carry
    is left out, never filled in.
    """
    attested = record.get("attested")
    attested = attested if isinstance(attested, Mapping) else {}
    pools = record.get("pools")
    pools = pools if isinstance(pools, Mapping) else {}
    memory = record.get("memory")
    memory = memory if isinstance(memory, Mapping) else {}
    out: dict[str, Any] = {}
    graph = attested.get("cuda_graph")
    captured = attested.get("cuda_graph_captured")
    if isinstance(captured, list):
        out["cuda_graph_batches"] = sorted({int(b) for b in captured})
    elif isinstance(graph, Mapping):
        out["cuda_graph_batches"] = sorted({int(b) for b in graph.get("batch_sizes") or ()})
    elif "cuda_graph" in attested:
        out["cuda_graph_batches"] = []
    if isinstance(graph, Mapping):
        out["cuda_graph_padded"] = bool(graph.get("enable_padding"))
    slots = [s for s in (_int(v) for v in pools.get("recurrent_slots") or ()) if s is not None]
    if slots:
        out["recurrent_slots"] = slots
    kv = _int(pools.get("kv_max_tokens"))
    logged = [t for t in (_int(v) for v in pools.get("kv_log_tokens") or ()) if t is not None]
    if kv is None and logged:
        kv = max(logged)
    if kv is not None:
        out["kv_tokens"] = kv
    devices = [
        d
        for d in memory.get("devices") or ()
        if isinstance(d, Mapping) and isinstance(d.get("free_gib"), (int, float))
    ]
    if devices:
        least = min(devices, key=lambda d: float(d["free_gib"]))
        out["free_gib"] = round(float(least["free_gib"]), 2)
        if isinstance(least.get("total_gib"), (int, float)):
            out["total_gib"] = round(float(least["total_gib"]), 2)
    if "speculative" in attested:
        spec = attested["speculative"]
        out["speculative"] = (
            f"{spec.get('algorithm')} depth {spec.get('max_draft_len')}"
            if isinstance(spec, Mapping)
            else "none"
        )
    for name in ("tp", "ep", *_LIMITS):
        value = _int(attested.get(name))
        if value is not None:
            out[name] = value
    if isinstance(attested.get("attention_dp"), bool):
        out["attention_dp"] = attested["attention_dp"]
    return out


def load(workspace: Path) -> dict[str, Any]:
    """The facts on disk; empty when there are none or the file cannot be read."""
    try:
        data = yaml.safe_load(facts_path(workspace).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    return data if isinstance(data, dict) else {}


def _member_facts(engines: Sequence[Mapping[str, Any]], current: str) -> dict[str, Any]:
    """The facts across a member's engines on the ``current`` target, or across all, stale.

    An unestablished ``current`` (``""``) judges nothing: every engine is taken.
    """
    taken = [e for e in engines if e.get("marker", "") == current] if current else list(engines)
    stale = not taken
    taken = taken or list(engines)
    facts: dict[str, Any] = {
        "stale": stale,
        "engines": [f"{e['gate']}/{e['leg']}" if e.get("leg") else e["gate"] for e in taken],
    }
    latest = max(taken, key=lambda e: str(e.get("at", "")))
    if latest.get("tree"):
        facts["tree"] = latest["tree"]
    batches = [e["cuda_graph_batches"] for e in taken if "cuda_graph_batches" in e]
    if batches:
        facts["cuda_graph_batches"] = sorted({int(b) for found in batches for b in found})
    padded = {bool(e["cuda_graph_padded"]) for e in taken if "cuda_graph_padded" in e}
    if padded:
        facts["cuda_graph_padding"] = (
            "padded and unpadded"
            if len(padded) > 1
            else ("padded" if True in padded else "unpadded")
        )
    slots = [s for e in taken for s in e.get("recurrent_slots") or ()]
    if slots:
        facts["recurrent_slots_per_rank"] = max(slots)
    for name in ("kv_tokens", *_LIMITS):
        found = [e[name] for e in taken if name in e]
        if found:
            facts[name] = max(found)
    free = [e for e in taken if "free_gib" in e]
    if free:
        least = min(free, key=lambda e: e["free_gib"])
        facts["free_gib"] = least["free_gib"]
        if "total_gib" in least:
            facts["total_gib"] = least["total_gib"]
    speculative = sorted({e["speculative"] for e in taken if "speculative" in e})
    if speculative:
        facts["speculative"] = speculative
    for name in ("tp", "ep", "attention_dp"):
        found = [e for e in taken if name in e]
        if found:
            facts[name] = max(found, key=lambda e: str(e.get("at", "")))[name]
    return facts


def _most(members: Mapping[str, Mapping[str, Any]], name: str) -> dict[str, Any] | None:
    found = [(key, m["facts"][name]) for key, m in members.items() if name in m["facts"]]
    if not found:
        return None
    key, value = max(found, key=lambda item: item[1])
    return {"most": value, "at": key}


def _envelope(members: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Across every member: the bounds a kernel serves, and where each was measured."""
    envelope: dict[str, Any] = {
        "members": len(members),
        "stale": sorted(key for key, m in members.items() if m["facts"].get("stale")),
    }
    batches = sorted(
        {int(b) for m in members.values() for b in m["facts"].get("cuda_graph_batches") or ()}
    )
    if batches:
        envelope["cuda_graph_batches"] = batches
    unpadded = sorted(
        key
        for key, m in members.items()
        if "unpadded" in str(m["facts"].get("cuda_graph_padding", ""))
    )
    if unpadded:
        envelope["cuda_graph_unpadded_at"] = unpadded
    for name in ("recurrent_slots_per_rank", "kv_tokens", "ep", *_LIMITS):
        most = _most(members, name)
        if most is not None:
            envelope[name] = most
    free = [(key, m["facts"]) for key, m in members.items() if "free_gib" in m["facts"]]
    if free:
        key, facts = min(free, key=lambda item: item[1]["free_gib"])
        envelope["free_gib"] = {
            "least": facts["free_gib"],
            "of": facts.get("total_gib"),
            "at": key,
        }
    speculative = sorted(
        {s for m in members.values() for s in m["facts"].get("speculative") or ()} - {"none"}
    )
    if speculative:
        envelope["speculative"] = speculative
    dp = sorted(key for key, m in members.items() if m["facts"].get("attention_dp"))
    if dp:
        envelope["attention_dp_at"] = dp
    return envelope


def _show(name: str, value: Any) -> str:
    if name == "cuda_graph_batches":
        return f"[{format_batches(value)}]" if value else "none"
    if name == "cuda_graph_padded":
        return "padded" if value else "unpadded"
    if name == "recurrent_slots":
        return str(max(value)) if value else "none"
    return str(value)


_CHANGE_NAMES = {
    "cuda_graph_batches": "CUDA-graph batches",
    "cuda_graph_padded": "CUDA-graph padding",
    "recurrent_slots": "recurrent slots per rank",
    "speculative": "speculation",
}


def _changes(before: Mapping[str, Any], after: Mapping[str, Any]) -> list[str]:
    """How one gate's engine moved between two passes, field by field."""
    out = []
    for name in CHANGE_FIELDS:
        if name not in before or name not in after:
            continue
        was, now = _show(name, before[name]), _show(name, after[name])
        if was != now:
            out.append(f"{_CHANGE_NAMES.get(name, name)} {was} -> {now}")
    return out


def record(
    workspace: Path,
    *,
    unit: str,
    member: str,
    gate: str,
    engines: Iterable[Mapping[str, Any]],
    tree: str = "",
    marker: str = "",
    current: str = "",
) -> list[str]:
    """Keep a passed gate's ``engines`` for ``unit``'s ``member`` and rewrite the file.

    ``marker`` is the target the gate measured, ``current`` the target as
    it is now: a member's facts are taken across its engines on ``current``.
    Returns one line per engine whose numbers moved since the gate's last
    pass, as kept in ``changes``; an engine record with no numbers is left
    out.
    """
    at = _now()
    records = [r for r in engines if isinstance(r, Mapping)]
    measured = []
    for index, engine in enumerate(records):
        numbers = engine_numbers(engine)
        if not numbers:
            continue
        leg = str(engine.get("leg") or "") or (f"#{index + 1}" if len(records) > 1 else "")
        entry: dict[str, Any] = {"gate": gate}
        if leg:
            entry["leg"] = leg
        entry.update(at=at, tree=tree, marker=marker, attested=bool(engine.get("reported")))
        entry.update(numbers)
        measured.append(entry)
    if not measured:
        return []
    key = f"{unit}@{member}"
    with _LOCK:
        data = load(workspace)
        kept = data.get("members") if isinstance(data.get("members"), dict) else {}
        members = {
            str(k): {
                "unit": v.get("unit"),
                "member": v.get("member"),
                "engines": [e for e in v["engines"] if isinstance(e, dict) and e.get("gate")],
            }
            for k, v in kept.items()
            if isinstance(v, dict) and isinstance(v.get("engines"), list)
        }
        previous = members.get(key, {}).get("engines", [])
        changes: list[str] = []
        for entry in measured:
            same = next(
                (
                    e
                    for e in previous
                    if e["gate"] == entry["gate"] and e.get("leg", "") == entry.get("leg", "")
                ),
                None,
            )
            leg = f"/{entry['leg']}" if entry.get("leg") else ""
            if same is not None:
                moved = _changes(same, entry)
                if moved:
                    changes.append(f"{unit}:{gate}@{member}{leg}: {'; '.join(moved)}")
                previous.remove(same)
            previous.append(entry)
        members[key] = {"unit": unit, "member": member, "engines": previous}
        members = {k: v for k, v in members.items() if v["engines"]}
        for item in members.values():
            item["facts"] = _member_facts(item["engines"], current)
        history = data.get("changes") if isinstance(data.get("changes"), list) else []
        history = [str(c) for c in history]
        history += [f"{at} {line}" for line in changes]
        data = {
            "updated_at": at,
            "marker": current,
            "envelope": _envelope(members),
            "members": dict(sorted(members.items())),
            "changes": history[-MAX_CHANGES:],
        }
        _write(workspace, data)
    return changes


def _write(workspace: Path, data: Mapping[str, Any]) -> None:
    """Replace the file whole, so a reader never sees half of it."""
    path = facts_path(workspace)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        "# What the engines of passed gates ran, read back from each engine once it was up.\n"
        "# Written by the orchestrator (run/runtime_facts.py).\n"
        + yaml.dump(dict(data), Dumper=_Dumper, sort_keys=False, width=100),
        encoding="utf-8",
    )
    os.replace(partial, path)


def _gib(facts: Mapping[str, Any]) -> str:
    total = facts.get("total_gib")
    return f"{facts['free_gib']:.2f}" + (f" of {total:.2f}" if total is not None else "")


def member_line(key: str, facts: Mapping[str, Any]) -> str:
    """One member's facts as a prompt line."""
    parts = _fact_parts(facts)
    where = ", ".join(facts.get("engines") or ())
    if facts.get("tree"):
        where += f"; tree {facts['tree']}"
    stale = " STALE: measured on an earlier state of the target" if facts.get("stale") else ""
    return f"- {key}: {'; '.join(parts) or 'no numbers'} ({where}){stale}"


def _fact_parts(facts: Mapping[str, Any]) -> list[str]:
    """A member's facts, one phrase each."""
    parts = []
    if "cuda_graph_batches" in facts:
        batches = facts["cuda_graph_batches"]
        padding = f", {facts['cuda_graph_padding']}" if facts.get("cuda_graph_padding") else ""
        parts.append(
            f"{len(batches)} CUDA-graph batches [{format_batches(batches)}]{padding}"
            if batches
            else "no CUDA graphs"
        )
    if "recurrent_slots_per_rank" in facts:
        parts.append(f"{facts['recurrent_slots_per_rank']} recurrent slots per rank")
    if "kv_tokens" in facts:
        parts.append(f"KV pool {facts['kv_tokens']} tokens")
    if "free_gib" in facts:
        parts.append(f"{_gib(facts)} GiB free once up")
    speculative = [s for s in facts.get("speculative") or () if s != "none"]
    if speculative:
        parts.append("speculation " + ", ".join(speculative))
    elif facts.get("speculative"):
        parts.append("no speculation")
    widths = [f"{name} {facts[name]}" for name in ("tp", "ep") if name in facts]
    if facts.get("attention_dp"):
        widths.append("attention DP")
    if widths:
        parts.append(" ".join(widths))
    limits = [f"{name} {facts[name]}" for name in _LIMITS if name in facts]
    if limits:
        parts.append(", ".join(limits))
    return parts


def _envelope_lines(envelope: Mapping[str, Any]) -> list[str]:
    lines = []
    batches = envelope.get("cuda_graph_batches")
    if batches:
        unpadded = envelope.get("cuda_graph_unpadded_at")
        lines.append(
            f"- CUDA-graph batches captured: {len(batches)} sizes [{format_batches(batches)}]"
            + (f"; unpadded at {', '.join(unpadded)}" if unpadded else "")
        )
    for name, what in (
        ("recurrent_slots_per_rank", "recurrent slots per rank: up to {most}"),
        ("kv_tokens", "KV pool: up to {most} tokens"),
        ("max_batch_size", "max_batch_size: up to {most}"),
        ("max_num_tokens", "max_num_tokens: up to {most}"),
        ("ep", "expert parallel: up to {most}"),
    ):
        most = envelope.get(name)
        if isinstance(most, Mapping):
            lines.append(f"- {what.format(**most)} ({most.get('at')})")
    if envelope.get("speculative"):
        lines.append(f"- speculation: {', '.join(envelope['speculative'])}")
    if envelope.get("attention_dp_at"):
        lines.append(f"- attention DP at {', '.join(envelope['attention_dp_at'])}")
    free = envelope.get("free_gib")
    if isinstance(free, Mapping):
        of = f" of {free['of']:.2f}" if free.get("of") is not None else ""
        lines.append(f"- least memory free once up: {free['least']:.2f}{of} GiB ({free.get('at')})")
    return lines


def render(
    workspace: Path,
    *,
    unit: str | None = None,
    changes: int = 0,
) -> str:
    """The block a prompt carries: the file, the envelope, a line per member, the latest changes.

    ``unit``'s members come first. ``changes`` is how many of the latest
    changes to list. Empty when the file holds no member.
    """
    data = load(workspace)
    kept = data.get("members") if isinstance(data.get("members"), dict) else {}
    kept = {
        k: v for k, v in kept.items() if isinstance(v, dict) and isinstance(v.get("facts"), dict)
    }
    if not kept:
        return ""
    envelope = data.get("envelope") if isinstance(data.get("envelope"), dict) else {}
    stale = envelope.get("stale") or []
    lines = [
        HEADING,
        "",
        f"`{facts_path(workspace)}` (updated {data.get('updated_at')}) holds what the engines "
        "of passed gates ran, read back from each engine once it was up: per unit and member, "
        "the CUDA-graph batches captured, the recurrent-state slots per rank, the KV-cache "
        "pool, the memory left free, the speculation depth, the widths and the limits. Size "
        "and certify against these measured values, not estimates. "
        + (
            f"{len(stale)} of {len(kept)} members were last measured on an earlier state of "
            "the target."
            if stale
            else f"All {len(kept)} members were measured on the target as it was then."
        ),
    ]
    envelope_lines = _envelope_lines(envelope)
    if envelope_lines:
        lines += ["", f"Across every member ({len(kept)}):", *envelope_lines]
    order = sorted(kept, key=lambda k: (kept[k].get("unit") != unit, k))
    shown = order[:MEMBER_LINES]
    lines += ["", "Per member:", *(member_line(k, kept[k]["facts"]) for k in shown)]
    if len(order) > len(shown):
        lines.append(f"- and {len(order) - len(shown)} more members in the file")
    recent = [str(c) for c in data.get("changes") or ()][-changes:] if changes > 0 else []
    if recent:
        lines += ["", "Measured changes, newest last:", *(f"- {c}" for c in recent)]
    return "\n".join(lines)


__all__ = [
    "CHANGE_FIELDS",
    "FACTS_FILENAME",
    "HEADING",
    "MAX_CHANGES",
    "MEMBER_LINES",
    "engine_numbers",
    "facts_path",
    "load",
    "member_line",
    "record",
    "render",
]
