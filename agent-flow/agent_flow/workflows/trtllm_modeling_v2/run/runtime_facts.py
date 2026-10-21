# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime facts: what the engines of passed gates ran, per unit and member.

Every gate that builds an engine attests it (``GateResult.engine``, read back
once the engine is up by ``checks/harness/engine_facts.py`` and
``gates.attest_engine``): the CUDA-graph batches it captured and whether it
pads to them, the recurrent-state slots each rank serves from, its KV-cache
pool in tokens, the device memory it left free, its speculation depth, its
widths and its batch and token limits. Those are the values a kernel serves.
A cell certified at an estimate instead (a slot count, a batch the engine
does not pad, a verify path under capture) is certified again once the true
value shows, so the measured ones are kept here, in ``runtime_facts.yaml``,
for certification and sizing to start from.

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
  limits), newest last; a gate's first pass is held against the geometry
  probe's engine for its unit and member, when the probe measured one. Free
  memory and the KV pool follow the device's other tenants and are not
  changes. A re-certification follows a change here.
- ``geometry``: what the geometry probe (``run/geometry.py``) measured before
  any kernel was certified, one engine boot per width under the run's runtime
  block: per leg (``<unit>@<member>``) the engine's numbers, the KV page view,
  the context chunks prompts of the probe lengths ran in, and the speculation
  depths the spec declares. :func:`record_geometry` writes it whole; the
  gates' records leave it as it is.

:func:`render` is the block a prompt carries: the path, the envelope, one
line per member, and the serving geometry.
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
#: The heading a prompt carries the geometry probe's measurement under.
GEOMETRY_HEADING = "## Serving geometry (measured before kernels)"
#: The key of the geometry probe's section.
GEOMETRY = "geometry"
#: A geometry leg's ``status`` when its engine came up and was read.
MEASURED = "measured"
#: A geometry leg's ``source`` when a stand-in checkpoint the stock path supports was booted.
SOURCE_PROXY = "proxy"

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
    at: str | None = None,
) -> list[str]:
    """Keep a passed gate's ``engines`` for ``unit``'s ``member`` and rewrite the file.

    ``marker`` is the target the gate measured, ``current`` the target as
    it is now: a member's facts are taken across its engines on ``current``.
    Returns one line per engine whose numbers moved since the gate's last
    pass, as kept in ``changes``; an engine record with no numbers is left
    out.
    """
    at = at or _now()
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
        probed = _probed_numbers(data, key)
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
            elif probed:
                moved = _changes(probed, entry)
                if moved:
                    changes.append(
                        f"{unit}:{gate}@{member}{leg}: differs from the geometry probe: "
                        + "; ".join(moved)
                    )
            previous.append(entry)
        members[key] = {"unit": unit, "member": member, "engines": previous}
        members = {k: v for k, v in members.items() if v["engines"]}
        for item in members.values():
            item["facts"] = _member_facts(item["engines"], current)
        history = data.get("changes") if isinstance(data.get("changes"), list) else []
        history = [str(c) for c in history]
        history += [f"{at} {line}" for line in changes]
        kept_geometry = data.get(GEOMETRY)
        data = {
            "updated_at": at,
            "marker": current,
            "envelope": _envelope(members),
            "members": dict(sorted(members.items())),
            "changes": history[-MAX_CHANGES:],
        }
        if isinstance(kept_geometry, dict):
            data[GEOMETRY] = kept_geometry
        _write(workspace, data)
    return changes


def _write(workspace: Path, data: Mapping[str, Any]) -> None:
    """Replace the file whole, so a reader never sees half of it."""
    path = facts_path(workspace)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        "# What the engines of passed gates ran, read back from each engine once it was up,\n"
        "# and the serving geometry the probe measured before any kernel was certified.\n"
        "# Written by the orchestrator (run/runtime_facts.py, run/geometry.py).\n"
        + yaml.dump(dict(data), Dumper=_Dumper, sort_keys=False, width=100),
        encoding="utf-8",
    )
    os.replace(partial, path)


def _probed_numbers(data: Mapping[str, Any], key: str) -> dict[str, Any]:
    """The numbers the geometry probe measured for ``<unit>@<member>``; empty when none."""
    geometry = data.get(GEOMETRY)
    legs = geometry.get("legs") if isinstance(geometry, Mapping) else None
    leg = legs.get(key) if isinstance(legs, Mapping) else None
    if not isinstance(leg, Mapping) or leg.get("status") != MEASURED:
        return {}
    numbers = leg.get("numbers")
    return dict(numbers) if isinstance(numbers, Mapping) else {}


def record_geometry(
    workspace: Path,
    *,
    legs: Iterable[Mapping[str, Any]],
    inputs: str,
    mode: str,
    tree: str = "",
    marker: str = "",
    verify_depths: Mapping[str, int] | None = None,
    at: str | None = None,
    merge: bool = False,
) -> dict[str, Any]:
    """Keep the geometry probe's ``legs`` as the file's ``geometry`` and rewrite it.

    Each leg is ``{"unit", "member", "source", "status", "detail", "numbers",
    "page", "chunks", "errors"}`` (``run/geometry.py``); ``inputs`` is the
    digest of what the probe booted, by which a resumed run knows the
    measurement still stands. With ``merge`` the legs replace only the legs
    of the same key, and the others stay: a probe of the target at one width
    beside the earlier probe of the rest. The members, envelope and changes
    are left as they are. Returns the section written.
    """
    at = at or _now()
    section: dict[str, Any] = {
        "measured_at": at,
        "inputs": inputs,
        "mode": mode,
        "tree": tree,
        "marker": marker,
        "legs": {
            f"{leg['unit']}@{leg['member']}": {
                k: v for k, v in leg.items() if v not in (None, {}, "")
            }
            for leg in legs
            if isinstance(leg, Mapping) and leg.get("unit") and leg.get("member")
        },
    }
    if verify_depths:
        section["verify_depths"] = {str(k): int(v) for k, v in sorted(verify_depths.items())}
    with _LOCK:
        data = load(workspace)
        earlier = data.get(GEOMETRY) if isinstance(data.get(GEOMETRY), Mapping) else {}
        if merge and isinstance(earlier.get("legs"), Mapping):
            section["legs"] = {**dict(earlier["legs"]), **section["legs"]}
        data[GEOMETRY] = section
        data.setdefault("updated_at", at)
        _write(workspace, data)
    return section


def geometry(workspace: Path) -> dict[str, Any]:
    """The geometry probe's section of the file; empty when the probe has not written one."""
    section = load(workspace).get(GEOMETRY)
    return section if isinstance(section, dict) else {}


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
    """A member's facts, or a geometry leg's (:func:`_leg_facts`), one phrase each."""
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
    members: int = MEMBER_LINES,
) -> str:
    """The block a prompt carries: the file, the envelope, a line per member, the geometry.

    ``unit``'s members come first. ``changes`` is how many of the latest
    changes to list. Empty when the file holds neither a member nor the
    geometry probe's measurement.
    """
    data = load(workspace)
    blocks = (
        _render_members(data, workspace, unit=unit, changes=changes, members=members),
        _render_geometry(data, workspace),
    )
    return "\n\n".join(block for block in blocks if block)


def _render_members(
    data: Mapping[str, Any], workspace: Path, *, unit: str | None, changes: int, members: int
) -> str:
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
    shown = order[: max(members, 0)]
    lines += ["", "Per member:", *(member_line(k, kept[k]["facts"]) for k in shown)]
    if len(order) > len(shown):
        lines.append(f"- and {len(order) - len(shown)} more members in the file")
    recent = [str(c) for c in data.get("changes") or ()][-changes:] if changes > 0 else []
    if recent:
        lines += ["", "Measured changes, newest last:", *(f"- {c}" for c in recent)]
    return "\n".join(lines)


def _leg_facts(leg: Mapping[str, Any]) -> dict[str, Any]:
    """A geometry leg's engine numbers (:func:`engine_numbers`) under a member's fact names."""
    numbers = leg.get("numbers") if isinstance(leg.get("numbers"), Mapping) else {}
    facts = {k: v for k, v in numbers.items() if k not in ("recurrent_slots", "cuda_graph_padded")}
    if "cuda_graph_padded" in numbers:
        facts["cuda_graph_padding"] = "padded" if numbers["cuda_graph_padded"] else "unpadded"
    if numbers.get("recurrent_slots"):
        facts["recurrent_slots_per_rank"] = max(numbers["recurrent_slots"])
    if "speculative" in numbers:
        facts["speculative"] = [numbers["speculative"]]
    return facts


def _page_text(page: Mapping[str, Any]) -> str:
    per_block = page.get("tokens_per_block")
    if not per_block:
        return ""
    text = f"KV pages of {per_block} tokens"
    if page.get("blocks"):
        text += f", {page['blocks']} pages in the pool"
    if page.get("max_blocks_per_seq"):
        text += (
            f", up to {page['max_blocks_per_seq']} pages per sequence"
            f" (max_seq_len {page.get('max_seq_len')})"
        )
    return text


def _snapshot_text(page: Mapping[str, Any]) -> str:
    policy = page.get("snapshot_policy")
    if not isinstance(policy, Mapping):
        return ""
    stated = [
        f"{name.replace('_', ' ')} {value}"
        for name, value in policy.items()
        if value not in (0, [], None)
    ]
    return "snapshot policy: " + ", ".join(stated) if stated else ""


def _chunk_text(chunks: Mapping[str, Any]) -> str:
    shown = [
        f"{length} -> [{', '.join(str(c) for c in sizes)}]"
        for length, sizes in sorted(chunks.items(), key=lambda item: int(item[0]))
        if isinstance(sizes, list)
    ]
    return "context chunks per prompt length: " + "; ".join(shown) if shown else ""


def geometry_leg_line(key: str, leg: Mapping[str, Any]) -> str:
    """One geometry leg as a prompt line."""
    proxy = leg.get("source") == SOURCE_PROXY
    label = f"{key}, on the stand-in checkpoint" if proxy else key
    if leg.get("status") != MEASURED:
        return f"- {label}: not measured: {leg.get('detail') or leg.get('status')}"
    page = leg.get("page") if isinstance(leg.get("page"), Mapping) else {}
    parts = _fact_parts(_leg_facts(leg))
    parts += [_page_text(page), _snapshot_text(page)]
    parts.append(_chunk_text(leg.get("chunks") if isinstance(leg.get("chunks"), Mapping) else {}))
    return f"- {label}: {'; '.join(p for p in parts if p) or 'no numbers'}"


def geometry_values(section: Mapping[str, Any]) -> dict[str, list[int]]:
    """The measured shapes across the geometry legs, as the values a queue row's cells state.

    ``decode_q_len`` is the tokens one request carries in a generation step:
    1, and 1 plus each speculation depth the spec declares.
    """
    legs = section.get("legs") if isinstance(section.get("legs"), Mapping) else {}
    measured = [
        leg for leg in legs.values() if isinstance(leg, Mapping) and leg.get("status") == MEASURED
    ]
    found: dict[str, set[int]] = {}

    def add(name: str, values: Iterable[Any]) -> None:
        for value in values:
            if isinstance(value, int) and not isinstance(value, bool):
                found.setdefault(name, set()).add(value)

    for leg in measured:
        numbers = leg.get("numbers") if isinstance(leg.get("numbers"), Mapping) else {}
        page = leg.get("page") if isinstance(leg.get("page"), Mapping) else {}
        chunks = leg.get("chunks") if isinstance(leg.get("chunks"), Mapping) else {}
        add("cuda_graph_batches", numbers.get("cuda_graph_batches") or ())
        add(
            "recurrent_slots_per_rank",
            [max(numbers["recurrent_slots"])] if numbers.get("recurrent_slots") else (),
        )
        add("kv_tokens_per_block", [page.get("tokens_per_block")])
        add("max_blocks_per_seq", [page.get("max_blocks_per_seq")])
        add("max_num_tokens", [numbers.get("max_num_tokens"), page.get("max_num_tokens")])
        add(
            "context_chunk_tokens",
            [c for sizes in chunks.values() if isinstance(sizes, list) for c in sizes],
        )
    depths = (
        section.get("verify_depths") if isinstance(section.get("verify_depths"), Mapping) else {}
    )
    add("decode_q_len", [1, *(1 + int(d) for d in depths.values() if isinstance(d, int))])
    return {name: sorted(values) for name, values in found.items()}


def render_geometry(workspace: Path) -> str:
    """The geometry probe's block alone; empty when the probe has not written one."""
    return _render_geometry(load(workspace), workspace)


def _render_geometry(data: Mapping[str, Any], workspace: Path) -> str:
    section = data.get(GEOMETRY)
    legs = section.get("legs") if isinstance(section, Mapping) else None
    if not isinstance(legs, Mapping) or not legs:
        return ""
    path = "the target" if section.get("mode") == "require" else "the stock path"
    lines = [
        GEOMETRY_HEADING,
        "",
        f"`{facts_path(workspace)}` `{GEOMETRY}` (measured {section.get('measured_at')}"
        + (f", tree {section['tree']}" if section.get("tree") else "")
        + f") holds what one engine boot per width under this run's runtime block measured "
        f"on {path}, before any kernel was certified: the CUDA-graph batches captured, the "
        "KV page view, the limits, and the context chunks prompts of each probe length ran "
        "in. Certify the shapes the engine serves, not estimates; a gate's own measurement "
        "above supersedes a leg here. A leg on the stand-in checkpoint measured this run's "
        "configuration on another model, so its pools are not this model's and are left "
        "out.",
        "",
        *(geometry_leg_line(str(k), v) for k, v in sorted(legs.items()) if isinstance(v, Mapping)),
    ]
    depths = (
        section.get("verify_depths") if isinstance(section.get("verify_depths"), Mapping) else {}
    )
    if depths:
        lines.append(
            "- speculation depths the spec declares: "
            + ", ".join(f"{k} {v} (decode q_len {1 + int(v)})" for k, v in depths.items())
        )
    values = geometry_values(section)
    if values:
        lines += [
            "",
            "The measured values a verify row's `cells` (or a certification task's ```cells "
            "block) state, each under the entry's own spec key:",
            *(
                f"- {name}: {format_batches(found) if name == 'cuda_graph_batches' else found}"
                for name, found in values.items()
            ),
        ]
    return "\n".join(lines)


__all__ = [
    "CHANGE_FIELDS",
    "FACTS_FILENAME",
    "GEOMETRY",
    "GEOMETRY_HEADING",
    "HEADING",
    "MAX_CHANGES",
    "MEASURED",
    "MEMBER_LINES",
    "SOURCE_PROXY",
    "engine_numbers",
    "facts_path",
    "geometry",
    "geometry_leg_line",
    "geometry_values",
    "load",
    "member_line",
    "record",
    "record_geometry",
    "render",
    "render_geometry",
]
