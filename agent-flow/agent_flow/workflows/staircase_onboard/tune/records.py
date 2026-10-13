# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Perf records: measurements are recorded, never gated.

A measurement is a label dir written by ``bench.py`` (``summary.json`` +
``meta.json``). This module reads them, compares two of them under a noise
floor, and appends every measurement and every decision to the campaign's
``perf_ledger.jsonl``. The ledger is append-only, like the gate ledger: a
re-measured label is a new record, never an overwrite.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

# The two headline metrics of a two-point curve: the latency end is judged on
# tok/s/user at the lowest concurrency, the throughput end on output tok/s/GPU
# at the highest. Both are "higher is better".
LATENCY_METRIC = "tok_s_user"
THROUGHPUT_METRIC = "output_tok_s_gpu"
# Secondary metrics a decision reports but does not judge on.
SECONDARY = ("mean_tpot_ms", "mean_ttft_ms", "p99_tpot_ms", "p99_ttft_ms")


@dataclass(frozen=True)
class Measurement:
    label: str
    path: Path
    points: Mapping[int, Mapping[str, Any]]
    meta: Mapping[str, Any]

    @classmethod
    def load(cls, path: Path) -> "Measurement":
        rows = json.loads((path / "summary.json").read_text())
        meta = json.loads((path / "meta.json").read_text())
        return cls(
            label=meta.get("label", path.name),
            path=path,
            points={int(r["concurrency"]): r for r in rows},
            meta=meta,
        )

    def headline(self) -> dict[str, float]:
        """``{"con1.tok_s_user": ..., "con128.output_tok_s_gpu": ...}`` for the curve's two ends."""
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
    repeats. With fewer than two repeats the floor is unknown and ``default``
    stands in -- a decision then says so."""

    spread: dict[str, float] = field(default_factory=dict)
    samples: int = 0
    default: float = 0.02

    @classmethod
    def from_measurements(cls, repeats: Iterable[Measurement], default: float = 0.02) -> "NoiseFloor":
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

    def of(self, metric: str) -> float:
        return self.spread.get(metric, self.default)


@dataclass(frozen=True)
class Comparison:
    candidate: str
    control: str
    deltas: Mapping[str, float]
    verdict: str  # WIN | LOSS | NEUTRAL
    reason: str

    def as_dict(self) -> dict:
        return {
            "candidate": self.candidate,
            "control": self.control,
            "deltas": dict(self.deltas),
            "verdict": self.verdict,
            "reason": self.reason,
        }


def compare(
    candidate: Measurement,
    control: Measurement,
    noise: NoiseFloor,
    *,
    min_gain: float = 0.03,
    noise_factor: float = 2.0,
) -> Comparison:
    """The keep rule: better by at least the threshold somewhere, worse beyond the noise nowhere.

    The threshold is ``max(min_gain, noise_factor * spread)`` per metric: a 3%
    gain on a metric whose repeats wander by 2.5% is not a gain."""
    cand, ctrl = candidate.headline(), control.headline()
    deltas = {m: relative(cand[m], ctrl[m]) for m in ctrl if m in cand}
    wins, losses = [], []
    for metric, delta in deltas.items():
        threshold = max(min_gain, noise_factor * noise.of(metric))
        floor = max(0.01, noise_factor * noise.of(metric))
        if delta >= threshold:
            wins.append(f"{metric} {delta:+.1%} (>= {threshold:.1%})")
        elif delta <= -floor:
            losses.append(f"{metric} {delta:+.1%} (<= -{floor:.1%})")
    if losses:
        verdict, reason = "LOSS", "; ".join(losses + wins)
    elif wins:
        verdict, reason = "WIN", "; ".join(wins)
    else:
        verdict, reason = "NEUTRAL", "every delta inside the noise-scaled band"
    return Comparison(candidate.label, control.label, deltas, verdict, reason)


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
            points={str(c): {k: r.get(k) for k in (LATENCY_METRIC, THROUGHPUT_METRIC, *SECONDARY)} for c, r in m.points.items()},
            commit=tree.get("commit"),
            dirty=bool(tree.get("dirty_files")),
            diff_sha256=tree.get("diff_sha256"),
            host=m.meta.get("hostname"),
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
    head = "| label | modeling_v2 | commit | " + " | ".join(
        ["con lo tok/s/user", "con lo TPOT ms", "con lo TTFT ms", "con hi tok/s/GPU", "con hi TPOT ms", "con hi TTFT ms", "vs baseline"]
    ) + " |"
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    base = baseline.headline() if baseline else {}
    for m in rows:
        lo, hi = min(m.points), max(m.points)
        h = m.headline()
        vs = ", ".join(f"{k.split('.')[0]} {relative(h[k], base[k]):+.1%}" for k in base if k in h and k.split(".")[1] in (LATENCY_METRIC, THROUGHPUT_METRIC) and not (k.endswith(LATENCY_METRIC) and k.startswith(f"con{hi}") and lo != hi))
        tree = m.meta.get("tree") or {}
        commit = (tree.get("commit") or "")[:10] + ("*" if tree.get("dirty_files") else "")
        p_lo, p_hi = m.points[lo], m.points[hi]
        lines.append(
            f"| {m.label} | {m.meta.get('modeling_v2')} | {commit} | {p_lo[LATENCY_METRIC]:.2f} | {p_lo['mean_tpot_ms']:.2f} | "
            f"{p_lo['mean_ttft_ms']:.0f} | {p_hi[THROUGHPUT_METRIC]:.1f} | {p_hi['mean_tpot_ms']:.2f} | {p_hi['mean_ttft_ms']:.0f} | {vs} |"
        )
    return "\n".join(lines)
