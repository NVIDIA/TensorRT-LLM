# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Speed-of-light ceilings for the campaign's headline metrics.

A tuner turn derives them (``sol.yaml``), following the toolkit's
``internal-perf-sol-analysis`` skill when the session has it: hardware peaks
from its datasheet calculator, latency constants measured on the compute
node, and the analytical ceiling of every headline metric the campaign
measures. This module decides whether the file is usable and what it says
about a measurement. It computes no ceiling itself.

``sol.yaml``::

    method: calculator          # calculator: the skill's peaks and sol_calc.py; reduced: by hand
    peaks: tune/sol/peaks.json  # the peaks file the calculator used (calculator only)
    ceilings:                   # one per headline metric of the baseline, same keys
      con1.tok_s_user: {value: 412.0, bound: memory, derivation: "<formula with numbers>"}
      con128.output_tok_s_gpu: {value: 9100.0, bound: compute, derivation: "..."}
    notes: <assumptions>

A ceiling is an upper bound: a measurement above it means the model is
wrong (a miscounted byte, an optimistic peak), and the file is refused.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from .records import Measurement

SOL_FILENAME = "sol.yaml"
METHODS = ("calculator", "reduced")
BOUNDS = ("compute", "memory", "comm", "launch")
#: The skill the SOL turn loads, and its plugin spelling.
SOL_SKILL = "internal-perf-sol-analysis"
SOL_SKILL_QUALIFIED = f"trtllm-agent-toolkit:{SOL_SKILL}"


@dataclass(frozen=True)
class Ceiling:
    value: float
    bound: str
    derivation: str


@dataclass(frozen=True)
class SolCeilings:
    method: str
    ceilings: Mapping[str, Ceiling]
    peaks: str = ""
    notes: str = ""
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def calculated(self) -> bool:
        """The ceilings came from the skill's calculator, not from hand derivation."""
        return self.method == "calculator"

    def pct_of_sol(self, measurement: Measurement) -> dict[str, float]:
        """Per headline metric with a ceiling, measured as a percentage of it."""
        return {
            key: round(100.0 * value / self.ceilings[key].value, 1)
            for key, value in measurement.headline().items()
            if key in self.ceilings and self.ceilings[key].value > 0
        }

    def reached(self, measurement: Measurement, fraction: float) -> bool:
        """Every headline metric with a ceiling is at least ``fraction`` of it."""
        pct = self.pct_of_sol(measurement)
        return bool(pct) and all(p >= 100.0 * fraction for p in pct.values())


def load_sol(
    path: Path, baseline: Measurement, workdir: Path
) -> tuple[SolCeilings | None, list[str]]:
    """``sol.yaml`` checked against the baseline it is meant to bound; the errors, batched."""
    if not Path(path).is_file():
        return None, [f"{path} was not written"]
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        return None, [f"{path} is not YAML: {exc}"]
    if not isinstance(raw, Mapping):
        return None, [f"{path}: a mapping with `method` and `ceilings`"]
    errors: list[str] = []
    method = str(raw.get("method", ""))
    if method not in METHODS:
        errors.append(f"method: one of {list(METHODS)}, not {method!r}")
    peaks = str(raw.get("peaks") or "")
    if method == "calculator":
        peaks_path = Path(peaks) if Path(peaks).is_absolute() else Path(workdir) / peaks
        if not peaks or not peaks_path.is_file():
            errors.append(f"peaks: the calculator's peaks file {peaks!r} does not exist")
    ceilings: dict[str, Ceiling] = {}
    given = raw.get("ceilings")
    if not isinstance(given, Mapping):
        errors.append("ceilings: a mapping from headline metric to {value, bound, derivation}")
        given = {}
    headline = baseline.headline()
    for key in headline:
        if key not in given:
            errors.append(f"ceilings.{key}: missing; every headline metric needs a ceiling")
    for key, entry in given.items():
        where = f"ceilings.{key}"
        if key not in headline:
            errors.append(f"{where}: not a headline metric of the baseline ({sorted(headline)})")
            continue
        if not isinstance(entry, Mapping):
            errors.append(f"{where}: a mapping {{value, bound, derivation}}")
            continue
        try:
            value = float(entry.get("value"))
        except (TypeError, ValueError):
            errors.append(f"{where}.value: a positive number")
            continue
        bound = str(entry.get("bound", ""))
        derivation = str(entry.get("derivation", "")).strip()
        problems = []
        if value <= 0:
            problems.append("value must be positive")
        if bound not in BOUNDS:
            problems.append(f"bound must be one of {list(BOUNDS)}")
        if not derivation:
            problems.append("derivation must show the formula with its numbers")
        if value > 0 and headline[key] > value:
            problems.append(
                f"the baseline measured {headline[key]:.1f}, above this ceiling of {value:.1f}: "
                "the model is wrong (a miscounted byte or FLOP, an optimistic peak)"
            )
        if problems:
            errors.append(f"{where}: " + "; ".join(problems))
        else:
            ceilings[key] = Ceiling(value, bound, derivation)
    if errors:
        return None, errors
    return SolCeilings(method, ceilings, peaks, str(raw.get("notes") or "")), []


def render_table(sol: SolCeilings, measurements: list[Measurement]) -> str:
    """A markdown table: each headline metric's ceiling, and each measurement's % of it."""
    labels = [m.label for m in measurements]
    pcts = [sol.pct_of_sol(m) for m in measurements]
    lines = [
        "| metric | SOL | bound | " + " | ".join(f"{label} % of SOL" for label in labels) + " |",
        "|---|---|---|" + "---|" * len(labels),
    ]
    for key, ceiling in sol.ceilings.items():
        cells = [f"{p[key]:.1f}" if key in p else "-" for p in pcts]
        lines.append(
            f"| {key} | {ceiling.value:.1f} | {ceiling.bound} | " + " | ".join(cells) + " |"
        )
    if not sol.calculated:
        lines.append("")
        lines.append("(reduced: derived by hand, not by the SOL calculator)")
    return "\n".join(lines)


def describe(sol: SolCeilings | None, baseline: Measurement | None) -> str:
    """The standings paragraph a roadmap or candidate request carries."""
    if sol is None or baseline is None:
        return ""
    return "## Speed of light\n\n" + render_table(sol, [baseline]) + "\n"


def as_state(sol: SolCeilings) -> dict[str, Any]:
    return {
        "method": sol.method,
        "peaks": sol.peaks,
        "ceilings": {k: vars(c) for k, c in sol.ceilings.items()},
    }


def from_state(data: Mapping[str, Any]) -> SolCeilings | None:
    if not data or "ceilings" not in data:
        return None
    return SolCeilings(
        method=str(data.get("method", "")),
        peaks=str(data.get("peaks", "")),
        ceilings={k: Ceiling(**v) for k, v in data["ceilings"].items()},
    )


__all__ = [
    "BOUNDS",
    "Ceiling",
    "METHODS",
    "SOL_FILENAME",
    "SOL_SKILL",
    "SOL_SKILL_QUALIFIED",
    "SolCeilings",
    "as_state",
    "describe",
    "from_state",
    "load_sol",
    "render_table",
]
