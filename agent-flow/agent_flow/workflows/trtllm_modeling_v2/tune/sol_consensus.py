# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One SOL from several independent analyses: what they agree on, and a ruling on the rest.

Each method (:data:`METHODS`) counts the necessary work its own way, so their mistakes are
not the same mistakes: ``analytic`` from the model's config and the workload alone, ``code``
from the target's forward as written, ``profile`` from what a profile of the baseline shows
runs (with the traffic the profiler counted as an upper bound). Each analysis is a
calculator output the stage has validated on its own (``sol.load_result``).

:func:`reconcile` is deterministic:

- hardware capacities must agree (within 1 %): they are sourced numbers, not estimates;
- a headline ceiling is recomputed with the agreed hardware; one below the measured baseline
  is impossible and left out; ceilings within ``tolerance`` of each other agree, and the
  highest of them is taken (the hardest to claim reached); a ceiling only one of several
  analyses models is unconfirmed;
- a region whose id several analyses share and whose work agrees is kept, with the larger
  bound (the smaller headroom claimed); necessary bytes or FLOPs above what a profiler counted
  for that region rule an analysis's region out.

What disagrees, and what is unconfirmed, is a dispute. One adjudication turn rules on the
disputes by choosing an analysis (or none): it cannot write new numbers. A dispute left
without a valid ruling is unknown: it is neither ranked on nor a goal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .sol import TRAFFIC_SLACK, calculate

#: The analyses ``perf.sol.methods`` may ask for, in the order they are reported.
METHODS = ("analytic", "code", "profile")
#: Ceilings (or region work) this close agree: max / min - 1.
TOLERANCE = 0.15
#: Hardware capacities are sourced numbers: they agree only this close.
HARDWARE_TOLERANCE = 0.01
CAPACITIES = ("compute_flops_s", "memory_bytes_s")


@dataclass
class Consensus:
    #: The calculator inputs the analyses agree on (and the rulings chose).
    inputs: dict
    #: Per item: each analysis's value, the status (agreed, ruled, disputed, unknown) and why.
    report: dict
    #: What an adjudicator must rule on: ``{item, values: {method: value}, why}``.
    disputes: list[dict] = field(default_factory=list)
    #: Headline keys with no ceiling, and why: for ``SolCeilings.unknown``.
    unknown: dict[str, str] = field(default_factory=dict)


def _spread(values: Mapping[str, float]) -> float:
    low, high = min(values.values()), max(values.values())
    return high / low - 1 if low > 0 else float("inf")


def _hardware(
    analyses: Mapping[str, Mapping], rulings: Mapping[str, str | None], report: dict
) -> tuple[dict, list[dict]]:
    hardware: dict[str, Any] = {}
    disputes, notes = [], []
    for key in CAPACITIES:
        item = f"hardware.{key}"
        entries = {
            m: a["inputs"]["hardware"][key]
            for m, a in analyses.items()
            if (a["inputs"].get("hardware") or {}).get(key) is not None
        }
        values = {m: float(e["value"]) for m, e in entries.items()}
        row: dict[str, Any] = {"values": values}
        report[item] = row
        if not values:
            row["status"] = "unknown"
            continue
        if item in rulings:
            chosen = rulings[item]
            row["status"] = "ruled"
        elif _spread(values) <= HARDWARE_TOLERANCE:
            chosen = min(values)  # the first method: they agree
            row["status"] = "agreed"
        else:
            row["status"] = "disputed"
            disputes.append({"item": item, "values": values, "why": "the capacities differ"})
            continue
        row["chosen"] = chosen
        if chosen is not None:
            hardware[key] = dict(entries[chosen])
            notes.append(
                f"{key} from {chosen}: {analyses[chosen]['inputs']['hardware']['assumptions']}"
            )
    if notes:
        hardware["assumptions"] = "; ".join(notes)
    return hardware, disputes


def _ceiling(hardware: Mapping, key: str, entry: Mapping) -> float | None:
    try:
        ceilings = calculate({"hardware": hardware, "headline": {key: entry}})["ceilings"]
    except ValueError:
        return None
    return ceilings[key]["value"] if key in ceilings else None


def reconcile(
    analyses: Mapping[str, Mapping[str, Any]],
    baseline: Mapping[str, float],
    *,
    tolerance: float = TOLERANCE,
    rulings: Mapping[str, str | None] | None = None,
) -> Consensus:
    """The SOL the analyses (``{method: calculator output}``) agree on, held to the baseline."""
    rulings = dict(rulings or {})
    report: dict[str, Any] = {"analyses": sorted(analyses, key=_order)}
    hardware, disputes = _hardware(analyses, rulings, report)
    settled = not disputes
    headline: dict[str, Any] = {}
    unknown: dict[str, str] = {}
    for key in baseline:
        item = f"headline.{key}"
        entries = {
            m: a["inputs"]["headline"][key]
            for m, a in analyses.items()
            if key in a["ceilings"] and key in (a["inputs"].get("headline") or {})
        }
        # With the hardware still disputed, each ceiling is its own analysis's.
        values = {
            m: (_ceiling(hardware, key, e) if settled else analyses[m]["ceilings"][key]["value"])
            for m, e in entries.items()
        }
        impossible = sorted(m for m, v in values.items() if v is not None and v < baseline[key])
        values = {m: v for m, v in values.items() if v is not None and m not in impossible}
        row: dict[str, Any] = {"values": values}
        if impossible:
            row["below_baseline"] = impossible
        report[item] = row
        if item in rulings and (rulings[item] is None or rulings[item] in values):
            chosen, row["status"] = rulings[item], "ruled"
        elif not values:
            row["status"] = "unknown"
            unknown[key] = "no analysis models it" + (
                f" ({', '.join(impossible)} below the measured baseline)" if impossible else ""
            )
            continue
        elif len(values) == 1 and len(analyses) > 1:
            (only,) = values
            row["status"], why = "disputed", f"unconfirmed: only {only} models it"
            disputes.append({"item": item, "values": values, "why": why})
            unknown[key] = why
            continue
        elif _spread(values) > tolerance:
            row["status"] = "disputed"
            why = f"the analyses differ by {_spread(values):.0%} (more than {tolerance:.0%})"
            disputes.append({"item": item, "values": values, "why": why})
            unknown[key] = why
            continue
        else:
            chosen, row["status"] = max(values, key=values.get), "agreed"
        row["chosen"] = chosen
        if chosen is None:
            unknown[key] = "ruled: no analysis's work model holds"
        else:
            headline[key] = _credited(chosen, entries[chosen])
    regions = _regions(analyses, hardware, tolerance, report) if settled else []
    inputs = {"hardware": hardware, "regions": regions, "headline": headline}
    return Consensus(inputs=inputs, report=report, disputes=disputes, unknown=unknown)


def _credited(method: str, entry: Mapping) -> dict:
    """An analysis's work entry, its assumptions naming the analysis."""
    return {**entry, "assumptions": f"{method}: {entry.get('assumptions') or ''}"}


def _order(method: str) -> tuple[int, str]:
    return (METHODS.index(method) if method in METHODS else len(METHODS), method)


def _regions(
    analyses: Mapping[str, Mapping], hardware: Mapping, tolerance: float, report: dict
) -> list[dict]:
    """The regions several analyses name alike and count alike; none is ruled on."""
    by_id: dict[str, dict[str, Mapping]] = {}
    for method, analysis in analyses.items():
        for entry in analysis["inputs"].get("regions") or []:
            by_id.setdefault(str(entry["id"]), {})[method] = entry
    kept, counts = [], {"agreed": 0, "single": 0, "disputed": 0, "over_counted": []}
    for region_id, entries in sorted(by_id.items()):
        counted = {
            work: min((float(e[f"measured_{work}"]) for e in entries.values()
                       if e.get(f"measured_{work}") is not None), default=None)
            for work in ("bytes", "flops")
        }  # fmt: skip
        over = sorted(
            m
            for m, e in entries.items()
            for work, cap in counted.items()
            if cap is not None and e.get(work) is not None
            and float(e[work]) > cap * (1 + TRAFFIC_SLACK)
        )  # fmt: skip
        if over:
            counts["over_counted"].append({"id": region_id, "analyses": over})
        entries = {m: e for m, e in entries.items() if m not in over}
        if len(entries) < 2:
            counts["single"] += 1
            continue
        agree = all(
            len(vals) < 2 or _spread(vals) <= tolerance
            for vals in (
                {m: float(e[work]) for m, e in entries.items() if e.get(work) is not None}
                for work in ("flops", "bytes")
            )
        )
        rows = {}
        for method, entry in entries.items():
            try:
                rows[method] = calculate({"hardware": hardware, "regions": [entry]})["regions"][0]
            except ValueError:
                continue  # its measured time is under the bound the agreed hardware sets
        bounded = {m: r["lower_bound_ms"] for m, r in rows.items() if r["lower_bound_ms"]}
        if not agree or not bounded:
            counts["disputed"] += 1
            continue
        counts["agreed"] += 1
        chosen = max(bounded, key=bounded.get)
        kept.append(_credited(chosen, entries[chosen]))
    report["regions"] = counts
    return kept


def read_rulings(data: Any, disputes: list[dict]) -> tuple[dict[str, str | None], list[str]]:
    """``rulings.yaml``'s rulings on ``disputes``: an analysis (or null) with its evidence."""
    asked = {d["item"]: d["values"] for d in disputes}
    rulings: dict[str, str | None] = {}
    problems = []
    rows = (data or {}).get("rulings") if isinstance(data, Mapping) else None
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, Mapping) or row.get("item") not in asked:
            problems.append(f"not a dispute: {row!r:.120}")
            continue
        choice = row.get("analysis")
        if choice is not None and choice not in asked[row["item"]]:
            problems.append(f"{row['item']}: {choice!r} is not one of {sorted(asked[row['item']])}")
        elif not str(row.get("evidence") or "").strip():
            problems.append(f"{row['item']}: a ruling needs its evidence")
        else:
            rulings[row["item"]] = choice
    return rulings, problems


__all__ = ["METHODS", "TOLERANCE", "Consensus", "read_rulings", "reconcile"]
