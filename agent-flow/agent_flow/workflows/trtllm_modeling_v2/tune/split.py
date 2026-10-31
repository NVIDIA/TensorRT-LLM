# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A LOSS split by member: the losses share one parallel mode, the wins lie outside it.

A *site* is one member (perf config) at one concurrency. When every site that lost
shares a parallel mode that no site that won is in, the LOSS is a ``split``: the same
change behind a condition the target knows -- attention DP on or off, an EP, TP or
world size (its ``mapping``), or the requests a rank runs in a step -- leaves that
mode on the old path and keeps the gain everywhere else.

:func:`classify` finds the mode: one trait, else two together, and among those
the one that covers the fewest sites. :func:`judge_variant` is the verdict on
the variant a tuner builds for it.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Collection, Mapping

from .records import Comparison, Measurement, NoiseFloor, bands, member_of, relative

SPLIT, MIXED, WHOLE = "split", "mixed", "whole"

#: The traits a mode is made of, in the order that breaks a tie: what the mode
#: calls the trait, and what the target reads it from.
AXES = {
    "attention_dp": ("attention DP", "mapping.enable_attention_dp"),
    "ep_size": ("EP size", "mapping.moe_ep_size"),
    "tp_size": ("TP size", "mapping.tp_size"),
    "world_size": ("world size", "mapping.world_size"),
    "batch": ("requests per rank in a step", "the requests this rank runs in the step"),
}
#: The most traits one mode joins.
MAX_TERMS = 2

Term = tuple[str, str, Any]


def member_traits(meta: Mapping[str, Any]) -> dict[str, Any]:
    """One member's parallel traits, from the merged LLM-API config ``bench.py`` recorded."""
    config = meta.get("config") or {}

    def size(key: str) -> int:
        try:
            return max(1, int(config.get(key) or 1))
        except (TypeError, ValueError):
            return 1

    tp = size("tensor_parallel_size")
    world = tp * size("pipeline_parallel_size") * size("context_parallel_size")
    if config.get("moe_expert_parallel_size"):
        ep = size("moe_expert_parallel_size")
    elif config.get("moe_tensor_parallel_size"):
        ep = max(1, tp // size("moe_tensor_parallel_size"))
    else:
        ep = 1  # TensorRT-LLM's default: experts split by TP
    return {
        "attention_dp": bool(config.get("enable_attention_dp")),
        "ep_size": ep,
        "tp_size": tp,
        "world_size": world,
    }


def _site_of(key: str) -> tuple[str, int] | None:
    """``("dep4", 256)`` for ``dep4.con256.output_tok_s_gpu``; ``("", 1)`` for ``con1.tok_s_user``."""
    parts = key.rsplit(".", 2)
    if len(parts) < 2 or not parts[-2].startswith("con") or not parts[-2][3:].isdigit():
        return None
    return (parts[0] if len(parts) == 3 else ""), int(parts[-2][3:])


def _site_name(member: str, concurrency: int) -> str:
    return f"{member}.con{concurrency}" if member else f"con{concurrency}"


def _holds(term: Term, traits: Mapping[str, Any]) -> bool:
    axis, op, value = term
    have = traits[axis]
    return have == value if op == "==" else (have >= value if op == ">=" else have <= value)


def _describe(term: Term) -> str:
    """The mode one term names, as the journal and the request say it."""
    axis, op, value = term
    if axis == "attention_dp":
        return "attention DP " + ("on" if value else "off")
    if axis == "batch":
        return f"{'high' if op == '>=' else 'low'} concurrency ({op} {value} requests per rank in a step)"
    return f"{AXES[axis][0]} {value}" if op == "==" else f"{AXES[axis][0]} {op} {value}"


def _condition(term: Term) -> str:
    """What the target reads for one term."""
    axis, op, value = term
    if axis == "attention_dp":
        return ("" if value else "not ") + AXES[axis][1]
    return f"{AXES[axis][1]} {op} {value}"


@dataclass(frozen=True)
class Split:
    """How a LOSS's losses and wins lie across the members (:func:`classify`).

    ``kind`` is ``split`` (a mode holds every losing site and no winning one),
    ``mixed`` (wins and losses, no mode between them) or ``whole`` (no win).
    """

    kind: str
    #: Headline keys that lost / won, and the constraints broken that the control kept.
    losses: tuple[str, ...]
    wins: tuple[str, ...]
    broken: tuple[str, ...]
    #: Sites (``member.conN``) that lost / won.
    lost: tuple[str, ...]
    won: tuple[str, ...]
    terms: tuple[Term, ...] = ()
    #: Every site the mode covers: where a variant keeps the old path.
    region: tuple[str, ...] = ()
    why: str = ""

    @property
    def mode(self) -> str:
        return " and ".join(_describe(t) for t in self.terms)

    @property
    def condition(self) -> str:
        return " and ".join(_condition(t) for t in self.terms)

    @property
    def members(self) -> tuple[str, ...]:
        return tuple(sorted({s.rpartition(".con")[0] for s in self.lost} - {""}))

    def as_dict(self) -> dict[str, Any]:
        return {
            "shape": self.kind,
            "mode": self.mode,
            "condition": self.condition,
            "terms": [list(t) for t in self.terms],
            "members": list(self.members),
            "losses": list(self.losses),
            "wins": list(self.wins),
            "broken": list(self.broken),
            "lost": list(self.lost),
            "won": list(self.won),
            "region": list(self.region),
            "why": self.why,
        }


def _metas(m: Measurement) -> dict[str, Mapping[str, Any]]:
    return {name: part.meta for name, part in m.parts.items()} if m.parts else {"": m.meta}


def classify(
    candidate: Measurement,
    control: Measurement,
    noise: NoiseFloor,
    *,
    min_gain: float,
    noise_factor: float,
    inert: Collection[str] = (),
) -> Split:
    """Where a candidate's losses and wins lie, judged by the keep rule's own bands.

    A member in ``inert`` (one the candidate provably does not change) neither wins nor
    loses, as in :func:`records.compare`.
    """
    cand, ctrl = candidate.headline(), control.headline()
    metas, fallback = _metas(candidate), _metas(control)
    sites: dict[str, dict[str, Any]] = {}
    lost: set[str] = set()
    won: set[str] = set()
    losses, wins, broken = [], [], []

    def site(member: str, con: int) -> str:
        name = _site_name(member, con)
        if name not in sites:
            meta = metas.get(member) or {}
            traits = member_traits(meta if meta.get("config") else fallback.get(member) or meta)
            dp = traits["attention_dp"]
            traits["batch"] = math.ceil(con / traits["world_size"]) if dp else con
            sites[name] = traits
        return name

    for key in ctrl:
        where = _site_of(key)
        if key not in cand or where is None or member_of(key) in inert:
            continue
        name = site(*where)
        delta = relative(cand[key], ctrl[key])
        threshold, floor = bands(key, noise, min_gain=min_gain, noise_factor=noise_factor)
        if delta >= threshold:
            won.add(name)
            wins.append(key)
        elif delta <= -floor:
            lost.add(name)
            losses.append(key)
    kept = set(control.violations())
    for member, con, text in candidate.violation_points():
        if text not in kept and member not in inert:
            lost.add(site(member, con))
            broken.append(text)
    shape = dict(
        losses=tuple(losses), wins=tuple(wins), broken=tuple(broken),
        lost=tuple(sorted(lost)), won=tuple(sorted(won)),
    )  # fmt: skip
    if not won:
        return Split(WHOLE, **shape, why="it wins nowhere")
    if not lost:
        return Split(WHOLE, **shape, why="it loses nowhere")
    both = sorted(lost & won)
    if both:
        return Split(MIXED, **shape, why=f"{', '.join(both)} both won and lost")
    terms = _separator(sites, lost, won)
    if terms is None:
        return Split(
            MIXED, **shape,
            why="no parallel mode (attention DP, an EP, TP or world size, the requests per rank, "
            "or two of them) holds every loss and no win",
        )  # fmt: skip
    region = tuple(sorted(n for n, t in sites.items() if all(_holds(term, t) for term in terms)))
    return Split(SPLIT, **shape, terms=terms, region=region)


def _terms(sites: Mapping[str, Mapping[str, Any]], lost: set[str]) -> list[tuple[Term, frozenset]]:
    """Each one-trait term true at every losing site and false somewhere, with the sites it covers."""
    out = []
    for axis in AXES:
        values = {sites[s][axis] for s in lost}
        exact = [(axis, "==", next(iter(values)))] if len(values) == 1 and axis != "batch" else []
        ranges = (
            [] if axis == "attention_dp" else [(axis, ">=", min(values)), (axis, "<=", max(values))]
        )
        for term in exact + ranges:
            cover = frozenset(n for n, t in sites.items() if _holds(term, t))
            if cover != frozenset(sites):
                out.append((term, cover))
    return out


def _separator(
    sites: Mapping[str, Mapping[str, Any]], lost: set[str], won: set[str]
) -> tuple[Term, ...] | None:
    """The fewest terms (one per trait) that cover every losing site and no winning one.

    Among those, the narrowest (fewest sites covered); among equals, the first in ``AXES``.
    """
    terms = _terms(sites, lost)
    for n in range(1, MAX_TERMS + 1):
        best: tuple[tuple[Term, ...], frozenset] | None = None
        for combo in itertools.combinations(terms, n):
            if len({term[0] for term, _ in combo}) < n:
                continue
            cover = frozenset.intersection(*(c for _, c in combo))
            if not cover & won and (best is None or len(cover) < len(best[1])):
                best = (tuple(term for term, _ in combo), cover)
        if best is not None:
            return best[0]
    return None


def judge_variant(
    variant: Comparison,
    split: Mapping[str, Any],
    original: Mapping[str, float],
    noise: NoiseFloor,
    *,
    min_gain: float,
    noise_factor: float,
) -> tuple[bool, str]:
    """The verdict on a member-gated variant of a split candidate: ``(win, why)``.

    A WIN only when the keep rule calls it one -- a loss nowhere, so the members in
    the split's mode are now neutral -- and every point the candidate won still
    wins: at or above its threshold, or holding the candidate's gain within the
    loss floor (the tolerance a batch holds its members' gains to).
    """
    if variant.verdict != "WIN":
        return False, f"{variant.verdict}: {variant.reason}"
    gave = []
    for key in split.get("wins") or ():
        if key not in original:
            continue
        delta = variant.deltas.get(key)
        threshold, floor = bands(key, noise, min_gain=min_gain, noise_factor=noise_factor)
        if delta is None or delta < min(threshold, original[key] - floor):
            got = "unmeasured" if delta is None else f"{delta:+.1%}"
            gave.append(f"{key} {got} (the candidate {original[key]:+.1%})")
    if gave:
        return False, "gives back where the candidate won: " + "; ".join(gave)
    return True, f"neutral where {split.get('mode')}, every win of the candidate held"


__all__ = ["MIXED", "SPLIT", "WHOLE", "Split", "classify", "judge_variant", "member_traits"]
