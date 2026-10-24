# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Perf records: measurements are recorded, never gated.

A measurement is a label dir written by ``bench.py`` (``summary.json`` +
``meta.json``), or, for a campaign over several perf configs, a label dir
holding one such dir per config plus ``parts.json`` (the config names and
their latency constraints). This module reads them, compares two of them under a noise
floor, and appends every measurement and every decision to the campaign's
``perf_ledger.jsonl``. The ledger is append-only, like the gate ledger: a
re-measured label is a new record, never an overwrite.

A *site* is one perf config at one concurrency (``dep4.con256``; ``con1`` for a
single config); its *member* is the perf config (``dep4``; ``""``).

``tok_s_user`` is read on ``bench.py``'s basis, the output tokens over the
end-to-end latency summed over a point's requests. A row that records no basis
holds benchmark_serving's per-request mean there; it is restated from its own
throughput, duration and mean latency, so every label compares like for like.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Collection, Iterable, Mapping, Sequence

from .bench import USER_RATE_BASIS, user_rate

# The two headline metrics of a two-point curve: the latency end is judged on
# tok/s/user at the lowest concurrency, the throughput end on output tok/s/GPU
# at the highest. Both are "higher is better".
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
#: A LOSS whose every loss is within this many times its loss floor is measured again before
#: it counts: one disturbed run lands a loss there.
REMEASURE_WITHIN = 2.0


def site_of(metric: str) -> str:
    """``dep4.con256`` for ``dep4.con256.output_tok_s_gpu``; ``con1`` for ``con1.tok_s_user``."""
    return metric.rpartition(".")[0]


def member_of(metric: str) -> str:
    """``dep4`` for ``dep4.con256.output_tok_s_gpu``; ``""`` for a single config's metric."""
    return site_of(metric).rpartition(".")[0]


def on_user_rate_basis(row: Mapping[str, Any]) -> Mapping[str, Any]:
    """``row`` with its ``tok_s_user`` on ``bench.py``'s basis (module docstring).

    A row that records its basis is as written; an older one is restated when it has the
    fields to, its per-request mean kept as ``tok_s_user_per_request``.
    """
    if "tok_s_user_basis" in row:
        return row
    tokens = float(row.get("output_tok_s") or 0) * float(row.get("duration_s") or 0)
    rate = user_rate(tokens, row.get("completed") or 0, row.get("mean_e2el_ms") or 0)
    if rate is None:
        return row
    return {
        **row,
        LATENCY_METRIC: rate,
        "tok_s_user_basis": USER_RATE_BASIS,
        "tok_s_user_per_request": row.get(LATENCY_METRIC),
    }


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
        return cls(
            label=label,
            path=path,
            points={int(r["concurrency"]): on_user_rate_basis(r) for r in rows},
            meta=meta,
        )

    @classmethod
    def mean(cls, samples: Sequence["Measurement"]) -> "Measurement":
        """The point-by-point mean of measurements of one workload (same configs and points).

        Every numeric field of a point is averaged over the samples that have that point;
        the label joins theirs with ``+``; the meta (host, tree) is the first's.
        """
        samples = list(samples)
        if len(samples) == 1:
            return samples[0]
        first = samples[0]
        label = "+".join(m.label for m in samples)
        if first.parts:
            parts = {
                name: cls.mean([m.parts[name] for m in samples if name in m.parts])
                for name in first.parts
            }
            return cls(label=label, path=first.path, points={}, meta=first.meta, parts=parts,
                       constraints=first.constraints)  # fmt: skip
        points = {}
        for con, row in first.points.items():
            rows = [m.points[con] for m in samples if con in m.points]
            merged = dict(row)
            for key in row:
                values = [r.get(key) for r in rows]
                if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
                    merged[key] = statistics.fmean(values)
            points[con] = merged
        return cls(label=label, path=first.path, points=points, meta=first.meta)

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
        """``{"con1.tok_s_user": ..., "con128.output_tok_s_gpu": ...}`` for the curve's two ends.

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
        # The other metric at each end is informative and judged for regressions:
        # a change that buys throughput by wrecking latency at the same point is
        # visible here.
        out[f"con{hi}.{LATENCY_METRIC}"] = float(self.points[hi][LATENCY_METRIC] or 0.0)
        return out


def relative(new: float, old: float) -> float:
    return (new - old) / old if old else 0.0


@dataclass
class NoiseFloor:
    """Run-to-run spread of identical configurations, per headline metric.

    ``spread[m]`` is the largest relative deviation from the mean among the
    repeats. Both metrics of one site come from the same runs, so they share one
    spread, the wider: a point whose tok/s/user wanders by 4% wanders in tok/s/GPU
    as well, whatever two runs happened to read. With fewer than two repeats the
    floor is unknown and ``default`` stands in -- a decision then says so.
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
    def from_measurements(
        cls, repeats: Iterable[Measurement], default: float = 0.02
    ) -> "NoiseFloor":
        repeats = list(repeats)
        if len(repeats) < 2:
            return cls(samples=len(repeats), default=default)
        metrics = set.intersection(*(set(m.headline()) for m in repeats))
        spread = {}
        for metric in metrics:
            values = [m.headline()[metric] for m in repeats]
            mean = statistics.fmean(values)
            spread[metric] = max(abs(v - mean) / mean for v in values) if mean else 0.0
        return cls(spread=spread, samples=len(repeats), default=default)

    @classmethod
    def widest(cls, floors: Iterable["NoiseFloor"], default: float = 0.02) -> "NoiseFloor":
        """Per metric, the widest spread any of ``floors`` measured; ``samples`` is the fewest."""
        floors = [f for f in floors if f.spread]
        spread: dict[str, float] = {}
        for floor in floors:
            for metric, value in floor.spread.items():
                spread[metric] = max(spread.get(metric, 0.0), value)
        return cls(spread=spread, samples=min((f.samples for f in floors), default=0),
                   default=default)  # fmt: skip

    def of(self, metric: str) -> float:
        if metric in self.spread:
            return self.spread[metric]
        site = site_of(metric)
        shared = [value for m, value in self.spread.items() if site_of(m) == site]
        return max(shared) if shared else self.default


def _member_floor(runs: Mapping[str, Sequence[Measurement]], default: float) -> NoiseFloor:
    """The spread of each member's runs (two or more), keyed by its sites; ``""`` keys none."""
    spread: dict[str, float] = {}
    counts = []
    for member, samples in runs.items():
        if len(samples) < 2:
            continue
        floor = NoiseFloor.from_measurements(samples, default=default)
        counts.append(floor.samples)
        for metric, value in floor.spread.items():
            key = f"{member}.{metric}" if member else metric
            spread[key] = max(spread.get(key, 0.0), value)
    return NoiseFloor(spread=spread, samples=min(counts, default=0), default=default)


def ground_floors(
    repeats: Sequence[Measurement], default: float = 0.02
) -> tuple[NoiseFloor, NoiseFloor]:
    """``(same_host, cross_host)``: how repeats of one measurement spread, member by member.

    Each member's runs are grouped by the host they ran on (their meta's ``hostname``).
    The same-host floor is, per site, the widest spread one host's runs of a member read
    (two or more runs there); the cross-host spread is that of all a member's runs that
    ran on two hosts or more. A keep is judged against controls of its own host, so only
    the first is a floor; the second says how far apart the hosts read. A measurement of
    one config is its own member, ``""``.
    """
    by_host: dict[str, dict[str, list[Measurement]]] = {}
    hosts: dict[str, set[str]] = {}
    every: dict[str, list[Measurement]] = {}
    for m in repeats:
        for member, run in (m.parts or {"": m}).items():
            host = str(run.meta.get("hostname") or "")
            by_host.setdefault(host, {}).setdefault(member, []).append(run)
            hosts.setdefault(member, set()).add(host)
            every.setdefault(member, []).append(run)
    same = NoiseFloor.widest(
        [_member_floor(runs, default) for runs in by_host.values()], default=default
    )
    cross = _member_floor({k: v for k, v in every.items() if len(hosts[k]) > 1}, default)
    return same, cross


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
        }

    def judged(self) -> dict[str, float]:
        """The deltas the verdict rests on: every member's but the inert ones'."""
        return {k: v for k, v in self.deltas.items() if member_of(k) not in self.inert}


def bands(
    metric: str, noise: NoiseFloor, *, min_gain: float, noise_factor: float
) -> tuple[float, float]:
    """``(threshold, floor)`` of ``metric``: a gain at or above the one wins, a loss past the other loses."""
    spread = noise_factor * noise.of(metric)
    return max(min_gain, spread), max(0.01, spread)


def compare(
    candidate: Measurement,
    control: Measurement,
    noise: NoiseFloor,
    *,
    min_gain: float = 0.03,
    noise_factor: float = 2.0,
    inert: Collection[str] = (),
) -> Comparison:
    """The keep rule: better by at least the threshold somewhere, worse beyond the noise nowhere.

    The threshold is ``max(min_gain, noise_factor * spread)`` per metric: a 3%
    gain on a metric whose repeats wander by 2.5% is not a gain. A member in
    ``inert`` -- one the candidate provably does not change (:mod:`.inert`) -- is
    judged nowhere: its deltas are reported, and where it lost (or broke a
    constraint the control kept) is listed apart, in ``inert_losses``.
    """
    cand, ctrl = candidate.headline(), control.headline()
    deltas = {m: relative(cand[m], ctrl[m]) for m in ctrl if m in cand}
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
        inert_losses=tuple(dict.fromkeys(aside_sites)),
    )  # fmt: skip


def remeasure_reason(
    cmp: Comparison,
    noise: NoiseFloor,
    *,
    min_gain: float,
    noise_factor: float,
    within: float = REMEASURE_WITHIN,
) -> str:
    """Why ``cmp`` is measured once more before it counts; ``""`` when it stands as it is.

    A loss where the candidate changes nothing is the host's or the run's. A LOSS whose
    every loss is within ``within`` times its loss floor is where one disturbed run (a
    node still busy with the job before it) lands. A broken constraint, or any loss past
    that, stands.
    """
    if cmp.inert_losses:
        return "it lost where it changes nothing: " + ", ".join(cmp.inert_losses)
    if cmp.verdict != "LOSS" or cmp.broken or not cmp.losses:
        return ""
    near = []
    for key in cmp.losses:
        _, floor = bands(key, noise, min_gain=min_gain, noise_factor=noise_factor)
        if -cmp.deltas[key] > within * floor:
            return ""
        near.append(f"{key} {cmp.deltas[key]:+.1%} (floor {floor:.1%})")
    return f"every loss is within {within:g}x its floor: " + ", ".join(near)


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

    def entries(self, kind: str | None = None) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if kind is None or entry.get("kind") == kind:
                out.append(entry)
        return out


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
