# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Perf records: measurements are recorded, never gated.

A measurement is a label dir written by ``bench.py`` (``summary.json`` + ``meta.json``), or, over
several perf configs, a label dir holding one such dir per config plus ``parts.json`` (the config
names and their latency constraints). Every measurement and decision is appended to the
campaign's ``perf_ledger.jsonl``: a re-measured label is a new record, never an overwrite.

A *site* is one perf config at one concurrency (``dep4.con256``; ``con1`` for a single config);
its *member* is the perf config (``dep4``; ``""``).
"""

from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Collection, Iterable, Mapping

# The two headline metrics, both higher-is-better: tok/s/user at the lowest concurrency (the
# latency end) and output tok/s/GPU at the highest (the throughput end).
LATENCY_METRIC = "tok_s_user"
THROUGHPUT_METRIC = "output_tok_s_gpu"
# Secondary metrics a decision reports but does not judge on.
SECONDARY = (
    "mean_tpot_ms",
    "mean_ttft_ms",
    "p99_tpot_ms",
    "p99_ttft_ms",
    "tok_s_user_per_request",
)
PARTS_FILE = "parts.json"
#: A perf config's latency constraints, each an upper bound on a summary metric.
CONSTRAINT_METRICS = {"tpot_ms_max": "mean_tpot_ms", "ttft_ms_max": "mean_ttft_ms"}
#: The latency a keep must not lose: mean TTFT at concurrency 1, where it is one request's
#: prefill rather than its wait in a queue. Lower is better, so it is judged for losses
#: only, past a relative tolerance (``perf.campaign.ttft_tolerance``).
TTFT_METRIC = "mean_ttft_ms"
TTFT_TOLERANCE = 0.10


def site_of(metric: str) -> str:
    """``dep4.con256`` for ``dep4.con256.output_tok_s_gpu``; ``con1`` for ``con1.tok_s_user``."""
    return metric.rpartition(".")[0]


def member_of(metric: str) -> str:
    """``dep4`` for ``dep4.con256.output_tok_s_gpu``; ``""`` for a single config's metric."""
    return site_of(metric).rpartition(".")[0]


@dataclass(frozen=True)
class Measurement:
    label: str
    path: Path
    points: Mapping[int, Mapping[str, Any]]
    meta: Mapping[str, Any]
    #: Per perf config, for a measurement over several; empty for one curve.
    parts: Mapping[str, "Measurement"] = field(default_factory=dict)
    #: Per perf config, its constraints (``CONSTRAINT_METRICS`` keys).
    constraints: Mapping[str, Mapping[str, float]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "Measurement":
        meta = json.loads((path / "meta.json").read_text())
        label = meta.get("label", path.name)
        if (path / PARTS_FILE).exists():
            spec = json.loads((path / PARTS_FILE).read_text())
            return cls(
                label=label,
                path=path,
                points={},
                meta=meta,
                parts={name: cls.load(path / name) for name in spec["parts"]},
                constraints=spec.get("constraints") or {},
            )
        rows = json.loads((path / "summary.json").read_text())
        points = {int(r["concurrency"]): r for r in rows}
        return cls(label=label, path=path, points=points, meta=meta)

    def violations(self) -> list[str]:
        """Each constraint a point of this measurement breaks, as ``config@con: metric``."""
        return [text for _, _, text in self.violation_points()]

    def violation_points(self) -> list[tuple[str, int, str]]:
        """Each constraint a point breaks: ``(config, concurrency, "config@con: metric ...")``."""
        out = []
        for name, part in self.parts.items():
            for key, bound in (self.constraints.get(name) or {}).items():
                metric = CONSTRAINT_METRICS[key]
                for con, row in sorted(part.points.items()):
                    value = row.get(metric)
                    if value is not None and float(value) > float(bound):
                        out.append(
                            (name, con, f"{name}@con{con}: {metric} {float(value):.1f} > {bound}")
                        )
        return out

    def headline(self) -> dict[str, float]:
        """The curve's two ends, and the throughput of every point between them.

        ``{"con1.tok_s_user": ..., "con32.output_tok_s_gpu": ..., "con128.output_tok_s_gpu": ...}``.
        Over several perf configs, each config's headline prefixed by its name.
        """
        if self.parts:
            return {
                f"{name}.{metric}": value
                for name, part in self.parts.items()
                for metric, value in part.headline().items()
            }
        if not self.points:
            return {}
        lo, hi = min(self.points), max(self.points)
        out = {f"con{lo}.{LATENCY_METRIC}": float(self.points[lo][LATENCY_METRIC] or 0.0)}
        out[f"con{hi}.{THROUGHPUT_METRIC}"] = float(self.points[hi][THROUGHPUT_METRIC] or 0.0)
        if lo == hi:
            return out
        for c in sorted(self.points)[1:-1]:
            out[f"con{c}.{THROUGHPUT_METRIC}"] = float(self.points[c][THROUGHPUT_METRIC] or 0.0)
        # Judged for regressions too: a change that buys throughput by wrecking latency at the
        # same point shows here.
        out[f"con{hi}.{LATENCY_METRIC}"] = float(self.points[hi][LATENCY_METRIC] or 0.0)
        return out

    def ttft(self) -> dict[str, float]:
        """``{"con1.mean_ttft_ms": ...}``: mean TTFT at concurrency 1, per config that runs it."""
        if self.parts:
            return {
                f"{name}.{metric}": value
                for name, part in self.parts.items()
                for metric, value in part.ttft().items()
            }
        value = (self.points.get(1) or {}).get(TTFT_METRIC)
        return {} if value is None else {f"con1.{TTFT_METRIC}": float(value)}


def relative(new: float, old: float) -> float:
    return (new - old) / old if old else 0.0


@dataclass
class NoiseFloor:
    """Run-to-run spread of identical configurations, per headline metric.

    ``spread[m]`` is the largest relative deviation from the mean among the repeats; both metrics
    of one site come from the same runs, so they share the wider. With fewer than two repeats the
    floor is unknown and ``default`` stands in (a decision then says so).
    """

    spread: dict[str, float] = field(default_factory=dict)
    samples: int = 0
    default: float = 0.02

    def __post_init__(self) -> None:
        widest: dict[str, float] = {}
        for metric, value in self.spread.items():
            site = site_of(metric)
            widest[site] = max(widest.get(site, 0.0), float(value))
        self.spread = {metric: widest[site_of(metric)] for metric in self.spread}

    @classmethod
    def from_measurements(cls, repeats: Iterable[Measurement]) -> "NoiseFloor":
        repeats = list(repeats)
        if len(repeats) < 2:
            return cls(samples=len(repeats))
        metrics = set.intersection(*(set(m.headline()) for m in repeats))
        spread = {}
        for metric in metrics:
            values = [m.headline()[metric] for m in repeats]
            mean = statistics.fmean(values)
            spread[metric] = max(abs(v - mean) / mean for v in values) if mean else 0.0
        return cls(spread=spread, samples=len(repeats))

    @classmethod
    def widest(cls, floors: Iterable["NoiseFloor"]) -> "NoiseFloor":
        """Per metric, the widest spread any of ``floors`` measured; ``samples`` is the fewest."""
        floors = [f for f in floors if f.spread]
        spread: dict[str, float] = {}
        for floor in floors:
            for metric, value in floor.spread.items():
                spread[metric] = max(spread.get(metric, 0.0), value)
        return cls(spread=spread, samples=min((f.samples for f in floors), default=0))

    def of(self, metric: str) -> float:
        if metric in self.spread:
            return self.spread[metric]
        site = site_of(metric)
        shared = [value for m, value in self.spread.items() if site_of(m) == site]
        return max(shared) if shared else self.default


@dataclass(frozen=True)
class Comparison:
    candidate: str
    control: str
    deltas: Mapping[str, float]
    verdict: str  # WIN | LOSS | NEUTRAL
    reason: str
    #: Headline keys that lost beyond their floor, and the constraints broken that the control kept.
    losses: tuple[str, ...] = ()
    broken: tuple[str, ...] = ()
    #: Members the candidate provably does not change: judged nowhere.
    inert: tuple[str, ...] = ()
    #: Sites of those members that lost beyond their floor or broke a constraint.
    inert_losses: tuple[str, ...] = ()
    #: How much slower its first token came at concurrency 1, per config (``TTFT_METRIC``).
    ttft: Mapping[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "candidate": self.candidate,
            "control": self.control,
            "deltas": dict(self.deltas),
            "verdict": self.verdict,
            "reason": self.reason,
            "losses": list(self.losses),
            "broken": list(self.broken),
            "inert": list(self.inert),
            "inert_losses": list(self.inert_losses),
            "ttft": dict(self.ttft),
        }

    def judged(self) -> dict[str, float]:
        """The deltas the verdict rests on: every member's but the inert ones'."""
        return {k: v for k, v in self.deltas.items() if member_of(k) not in self.inert}


def bands(
    metric: str, noise: NoiseFloor, *, min_gain: float, noise_factor: float
) -> tuple[float, float]:
    """``(threshold, floor)``: a gain at or above the one wins, a loss past the other loses."""
    spread = noise_factor * noise.of(metric)
    return max(min_gain, spread), max(0.01, spread)


def compare(
    candidate: Measurement,
    control: Measurement,
    noise: NoiseFloor,
    *,
    min_gain: float,
    noise_factor: float,
    inert: Collection[str] = (),
    ttft_tolerance: float | None = TTFT_TOLERANCE,
) -> Comparison:
    """The keep rule: better by at least the threshold somewhere, worse beyond the noise nowhere.

    The threshold is ``max(min_gain, noise_factor * spread)`` per metric. A first token slower by
    more than ``ttft_tolerance`` at a config's concurrency 1 is a loss too (``None``: unjudged).
    A member in ``inert`` (:mod:`.inert`) is judged nowhere: where it lost or broke a constraint
    the control kept is listed apart, in ``inert_losses``.
    """
    cand, ctrl = candidate.headline(), control.headline()
    if (
        not ctrl
        or cand.keys() != ctrl.keys()
        or any(not math.isfinite(v) or v <= 0 for v in (*cand.values(), *ctrl.values()))
    ):
        return Comparison(
            candidate.label,
            control.label,
            {},
            "LOSS",
            "invalid or incomplete headline measurements",
        )
    deltas = {m: relative(cand[m], ctrl[m]) for m in ctrl}
    inert = frozenset(inert)
    wins, losses, lost, aside, aside_sites = [], [], [], [], []
    for metric, delta in deltas.items():
        threshold, floor = bands(metric, noise, min_gain=min_gain, noise_factor=noise_factor)
        if member_of(metric) in inert:
            if delta <= -floor:
                aside.append(f"{metric} {delta:+.1%}")
                aside_sites.append(site_of(metric))
            continue
        if delta >= threshold:
            wins.append(f"{metric} {delta:+.1%} (>= {threshold:.1%})")
        elif delta <= -floor:
            losses.append(f"{metric} {delta:+.1%} (<= -{floor:.1%})")
            lost.append(metric)
    slower = {}
    if ttft_tolerance is not None:
        cand_ttft, ctrl_ttft = candidate.ttft(), control.ttft()
        slower = {
            m: relative(cand_ttft[m], v) for m, v in ctrl_ttft.items() if m in cand_ttft and v
        }
    for metric, delta in slower.items():
        if delta <= ttft_tolerance:
            continue
        if member_of(metric) in inert:
            aside.append(f"{metric} {delta:+.1%}")
            aside_sites.append(site_of(metric))
        else:
            losses.append(f"{metric} {delta:+.1%} (> +{ttft_tolerance:.0%})")
            lost.append(metric)
    kept = set(control.violations())
    broken = []
    for member, con, text in candidate.violation_points():
        if text in kept:
            continue
        if member in inert:
            aside.append(f"breaks {text}")
            aside_sites.append(f"{member}.con{con}")
        else:
            broken.append(text)
    broken.sort()
    if broken:
        losses.append("breaks a constraint: " + ", ".join(broken))
    if losses:
        verdict, reason = "LOSS", "; ".join(losses + wins)
    elif wins:
        verdict, reason = "WIN", "; ".join(wins)
    else:
        verdict, reason = "NEUTRAL", "every delta inside the noise-scaled band"
    if inert:
        reason += f"; not judged at {', '.join(sorted(inert))}, where it changes nothing"
        reason += (": " + ", ".join(aside)) if aside else ""
    return Comparison(
        candidate.label, control.label, deltas, verdict, reason,
        losses=tuple(lost), broken=tuple(broken), inert=tuple(sorted(inert)),
        inert_losses=tuple(dict.fromkeys(aside_sites)), ttft=slower,
    )  # fmt: skip


def goal_reached(best: Measurement, reference: Measurement, ratio: float) -> bool:
    """Every headline metric of ``best`` is at least ``ratio`` x the reference's."""
    mine, theirs = best.headline(), reference.headline()
    shared = [m for m in theirs if m in mine]
    return bool(shared) and all(mine[m] >= ratio * theirs[m] for m in shared if theirs[m])


class PerfLedger:
    """Append-only JSONL of measurements and decisions for one campaign."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, kind: str, **record: Any) -> dict:
        entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "kind": kind, **record}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")
        return entry

    def measurement(self, m: Measurement, **extra: Any) -> dict:
        tree = m.meta.get("tree") or {}
        return self.append(
            "measurement",
            label=m.label,
            path=str(m.path),
            modeling_v2=m.meta.get("modeling_v2"),
            headline=m.headline(),
            points={
                str(c): {k: r.get(k) for k in (LATENCY_METRIC, THROUGHPUT_METRIC, *SECONDARY)}
                for c, r in m.points.items()
            },
            commit=tree.get("commit"),
            dirty=bool(tree.get("dirty_files")),
            diff_sha256=tree.get("diff_sha256"),
            host=m.meta.get("hostname"),
            violations=m.violations(),
            **extra,
        )


def format_table(measurements: Iterable[Measurement], baseline: Measurement | None = None) -> str:
    """A markdown curve table, one row per label, deltas against ``baseline``."""
    rows = list(measurements)
    if any(m.parts for m in rows):
        names = list(dict.fromkeys(n for m in rows for n in m.parts))
        base_parts = baseline.parts if baseline else {}
        return "\n\n".join(
            f"### {name}\n\n"
            + format_table(
                [replace(m.parts[name], label=m.label) for m in rows if name in m.parts],
                replace(base_parts[name], label=baseline.label) if name in base_parts else None,
            )
            for name in names
        )
    head = (
        "| label | modeling_v2 | commit | "
        + " | ".join(
            [
                "con lo tok/s/user",
                "con lo TPOT ms",
                "con lo TTFT ms",
                "con hi tok/s/GPU",
                "con hi TPOT ms",
                "con hi TTFT ms",
                "vs baseline",
            ]
        )
        + " |"
    )
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    base = baseline.headline() if baseline else {}
    for m in rows:
        lo, hi = min(m.points), max(m.points)
        h = m.headline()
        vs = ", ".join(
            f"{k.split('.')[0]} {relative(h[k], base[k]):+.1%}"
            for k in base
            if k in h
            and k.split(".")[1] in (LATENCY_METRIC, THROUGHPUT_METRIC)
            and not (k.endswith(LATENCY_METRIC) and k.startswith(f"con{hi}") and lo != hi)
        )
        tree = m.meta.get("tree") or {}
        commit = (tree.get("commit") or "")[:10] + ("*" if tree.get("dirty_files") else "")
        p_lo, p_hi = m.points[lo], m.points[hi]
        lines.append(
            f"| {m.label} | {m.meta.get('modeling_v2')} | {commit} | "
            f"{p_lo[LATENCY_METRIC]:.2f} | {p_lo['mean_tpot_ms']:.2f} | {p_lo['mean_ttft_ms']:.0f} | "
            f"{p_hi[THROUGHPUT_METRIC]:.1f} | {p_hi['mean_tpot_ms']:.2f} | {p_hi['mean_ttft_ms']:.0f} | "
            f"{vs} |"
        )
    return "\n".join(lines)
