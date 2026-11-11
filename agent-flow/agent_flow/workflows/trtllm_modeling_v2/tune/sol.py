# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public roofline analysis from explicit work and user-sourced capacities.

Run this file directly with ``--input work.yaml --output sol.yaml`` (CPU only).
No device specifications, efficiency factors, or model work counts are built in.
Regions are independent: nested/overlapping timings are never summed. Headline
rates require a separate full-interval work model, not a sum of region times.

Input YAML::

    hardware:
      compute_flops_s: null   # or {value: <FLOP/s>, source: <public citation>}
      memory_bytes_s: null    # or {value: <byte/s>, source: <public citation>}
      assumptions: "precision, dense/sparse mode, memory boundary, device/rank scope"
    regions:
      - id: decoder.layer0.projection/decode/shape-A/rank0
        flops: null
        bytes: null
        assumptions: "work formula, invocation count and traffic boundary"
        measured_ms: null
        measurement_source: "commit, workload, timing method and artifact"
        measured_bytes: null  # optional: the traffic a profiler counted (ncu), an upper bound
        measured_flops: null  # of the necessary work: bytes/flops may not exceed it
    headline: {}

Use explicit, sourced public capacities; source labels alone do not prove their
validity or publicness. Missing quantities are null, not zero. Work counts and
capacities must cover the same precision, devices, ranks and memory boundary.
The lower bound is 1000 * max(flops / compute_flops_s, bytes / memory_bytes_s).

Each optional headline entry is keyed by an exact baseline metric and supplies
its own flops, bytes, assumptions and output_tokens; output_tok_s_gpu also needs
gpu_count. Work must cover the entire metric interval. Per-user rates exclude
other users; throughput includes all required work and output tokens. Omit an
unsupported headline instead of extrapolating from decode-only or local timing.

Optional overhead {ms, source, assumption} must be exposed and nonoverlapping;
its bound is conditional and cannot establish a pct_of_sol early-stop goal.
The output embeds its inputs, regions and ceilings; consumers recompute it.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import yaml

if TYPE_CHECKING:
    from .records import Measurement

SOL_FILENAME = "sol.yaml"
BOUNDS = ("compute", "memory")
METHOD = "roofline"
#: Necessary work may exceed what a profiler counted by this much (its window, rounding).
TRAFFIC_SLACK = 0.1


def _number(value: Any, where: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where}: a finite {'positive' if positive else 'nonnegative'} number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{where}: number out of range") from exc
    if not math.isfinite(result) or result < 0 or (positive and result == 0):
        raise ValueError(f"{where}: a finite {'positive' if positive else 'nonnegative'} number")
    return result


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where}: a nonempty source or assumption is required")
    return value


def _mapping(value: Any, where: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise ValueError(f"{where}: a mapping is required")
    return value


def _capacity(hardware: Mapping, key: str) -> float | None:
    if hardware.get(key) is None:
        return None
    entry = _mapping(hardware[key], f"hardware.{key}")
    _text(entry.get("source"), f"hardware.{key}.source")
    return _number(entry.get("value"), f"hardware.{key}.value", positive=True)


def _bound(work: Mapping, capacities: tuple, where: str) -> dict:
    """max(FLOPs / FLOP/s, bytes / byte/s), plus explicitly conditional overhead."""
    times = {}
    missing = []
    for key, bound, capacity in zip(("flops", "bytes"), BOUNDS, capacities):
        amount = work.get(key)
        if amount is None:
            missing.append(key)
            continue
        amount = _number(amount, f"{where}.{key}")
        if amount and capacity is None:
            missing.append(f"hardware capacity for {key}")
        else:
            times[bound] = _number(
                1000 * (amount / capacity) if amount else 0,
                f"{where}.{bound}_ms",
                positive=amount > 0,
            )
    if work.get("flops") is not None or work.get("bytes") is not None:
        _text(work.get("assumptions"), f"{where}.assumptions")
    overhead = work.get("overhead")
    overhead_ms = 0.0
    if overhead is not None:
        overhead = _mapping(overhead, f"{where}.overhead")
        overhead_ms = _number(overhead.get("ms"), f"{where}.overhead.ms")
        _text(overhead.get("source"), f"{where}.overhead.source")
        _text(overhead.get("assumption"), f"{where}.overhead.assumption")
    result = {"lower_bound_ms": None, "bound": None, "conditional": overhead is not None}
    if missing:
        return {**result, "unknown": ", ".join(missing)}
    bound = max(times, key=times.get)
    return {
        **result,
        "lower_bound_ms": _number(times[bound] + overhead_ms, where),
        "bound": bound,
        "compute_ms": times["compute"],
        "memory_ms": times["memory"],
    }


def calculate(inputs: Mapping[str, Any]) -> dict:
    """Calculate independent regions and optional full-interval rate upper bounds.

    Capacities apply to exactly the same device/rank scope as the work counts;
    callers must document precision, traffic boundary, parallelism and sources.
    A source label is provenance, not proof that its content is public or correct.
    """
    inputs = _mapping(inputs, "inputs")
    hardware = _mapping(inputs.get("hardware", {}), "hardware")
    capacities = (_capacity(hardware, "compute_flops_s"), _capacity(hardware, "memory_bytes_s"))
    if any(c is not None for c in capacities):
        _text(hardware.get("assumptions"), "hardware.assumptions")
    given = inputs.get("regions", [])
    if not isinstance(given, list):
        raise ValueError("regions: a list is required")
    regions = []
    seen = set()
    for entry in given:
        entry = _mapping(entry, "region")
        region_id = _text(entry.get("id"), "region.id")
        if region_id in seen:
            raise ValueError(f"duplicate region id: {region_id}")
        seen.add(region_id)
        row = {
            "id": region_id,
            **_bound(entry, capacities, region_id),
            "measured_ms": None,
            "headroom_ms": None,
            "pct_of_sol": None,
        }
        if entry.get("measured_ms") is not None:
            measured = _number(entry["measured_ms"], f"{region_id}.measured_ms", positive=True)
            _text(entry.get("measurement_source"), f"{region_id}.measurement_source")
            row["measured_ms"] = measured
            lower = row["lower_bound_ms"]
            if lower is not None:
                if lower > measured:
                    raise ValueError(
                        f"{region_id}: lower bound exceeds measured time; fix scope/work/capacity"
                    )
                row["headroom_ms"] = measured - lower
                row["pct_of_sol"] = 100 * (lower / measured)
        for key, work in (("measured_bytes", "bytes"), ("measured_flops", "flops")):
            if entry.get(key) is None:
                continue
            counted = _number(entry[key], f"{region_id}.{key}", positive=True)
            _text(entry.get("measurement_source"), f"{region_id}.measurement_source")
            row[key] = counted
            if entry.get(work) is not None and float(entry[work]) > counted * (1 + TRAFFIC_SLACK):
                raise ValueError(
                    f"{region_id}: necessary {work} exceed what the profiler counted ({key}); "
                    "fix the work or its memory boundary"
                )
        regions.append(row)
    ceilings, unknown = {}, {}
    for key, entry in _mapping(inputs.get("headline", {}), "headline").items():
        if not isinstance(key, str) or not key.endswith((".tok_s_user", ".output_tok_s_gpu")):
            raise ValueError(f"headline.{key}: unsupported rate metric")
        entry = _mapping(entry, f"headline.{key}")
        result = _bound(entry, capacities, f"headline.{key}")
        tokens = entry.get("output_tokens")
        gpus = entry.get("gpu_count") if key.endswith(".output_tok_s_gpu") else 1
        if tokens is not None:
            tokens = _number(tokens, f"{key}.output_tokens", positive=True)
        if gpus is not None:
            gpus = _number(gpus, f"{key}.gpu_count", positive=True)
            if gpus < 1 or not gpus.is_integer():
                raise ValueError(f"{key}.gpu_count: a positive integer")
        lower = result["lower_bound_ms"]
        if result["conditional"]:
            unknown[key] = "conditional measured overhead cannot establish a SOL goal"
            continue
        if lower is None or lower == 0 or tokens is None or gpus is None:
            unknown[key] = result.get(
                "unknown", "positive interval bound, output_tokens and gpu_count required"
            )
            continue
        value = _number(((tokens / gpus) / lower) * 1000, key, positive=True)
        ceilings[key] = {
            "value": value,
            "bound": result["bound"],
            "derivation": f"({tokens} output tokens / {gpus} GPUs) / ({lower} ms / 1000); "
            + entry["assumptions"],
        }
    return {
        "method": METHOD,
        "inputs": dict(inputs),
        "ceilings": ceilings,
        "regions": regions,
        "unknown": unknown,
    }


@dataclass(frozen=True)
class Ceiling:
    value: float
    bound: str
    derivation: str


@dataclass(frozen=True)
class SolCeilings:
    method: str
    ceilings: Mapping[str, Ceiling]
    regions: list[dict] = field(default_factory=list)
    unknown: Mapping[str, str] = field(default_factory=dict)

    def pct_of_sol(self, measurement: Measurement) -> dict[str, float]:
        """Per headline metric with a ceiling, measured as a percentage of it."""
        return {
            key: round(100.0 * (value / self.ceilings[key].value), 1)
            for key, value in measurement.headline().items()
            if key in self.ceilings and math.isfinite(value) and value >= 0
        }

    def reached(self, measurement: Measurement, fraction: float) -> bool:
        """Unknown or contradicted ceilings cannot satisfy an early-stop goal."""
        headline = measurement.headline()
        return bool(headline) and all(
            key in self.ceilings
            and math.isfinite(value)
            and fraction * self.ceilings[key].value <= value <= self.ceilings[key].value
            for key, value in headline.items()
        )


def load_result(path: Path, baseline: Measurement) -> tuple[dict | None, list[str]]:
    """The calculator's output at ``path``, recomputed from its inputs and held to the baseline.

    Never trusts hand-entered numbers: an output its inputs do not reproduce, or a ceiling
    below what the baseline already measured, is refused.
    """
    if not Path(path).is_file():
        return None, [f"{path} was not written"]
    try:
        raw = _mapping(yaml.safe_load(Path(path).read_text(encoding="utf-8")), str(path))
        result = calculate(raw.get("inputs"))
        if raw != result:
            raise ValueError(
                "SOL output differs from calculation; rerun sol.py, do not edit ceilings"
            )
        headline = baseline.headline()
        for key, value in headline.items():
            _number(value, f"baseline.{key}", positive=True)
            if key in result["ceilings"] and value > result["ceilings"][key]["value"]:
                raise ValueError(f"{key}: baseline exceeds ceiling; fix scope/work/capacity")
        if set(result["ceilings"]) - set(headline):
            raise ValueError("ceilings contain metrics outside the baseline")
        return result, []
    except (OSError, ValueError, yaml.YAMLError) as exc:
        return None, [str(exc)]


def load_sol(path: Path, baseline: Measurement) -> tuple[SolCeilings | None, list[str]]:
    """Recompute CLI output from its inputs; never trust hand-entered headline numbers."""
    result, errors = load_result(path, baseline)
    if result is None:
        return None, errors
    unknown = dict(result["unknown"])
    for key in baseline.headline():
        if key not in result["ceilings"]:
            unknown.setdefault(key, "no full-interval work model")
    return SolCeilings(
        METHOD,
        {k: Ceiling(**v) for k, v in result["ceilings"].items()},
        regions=result["regions"],
        unknown=unknown,
    ), []


def render_table(sol: SolCeilings, measurements: list[Measurement]) -> str:
    """Headline bounds and independent module headroom; no region sum implies end-to-end gain."""
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
    for key, why in sol.unknown.items():
        lines.append(f"| {key} | unknown: {why} | - | " + " | ".join("-" for _ in labels) + " |")
    if sol.regions:
        lines += [
            "",
            "| region id | measured ms | bound ms | headroom ms | bound |",
            "|---|---|---|---|---|",
        ]
        for row in sol.regions:
            cells = [
                str(row.get(k)) if row.get(k) is not None else "unknown"
                for k in ("id", "measured_ms", "lower_bound_ms", "headroom_ms", "bound")
            ]
            lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "Public roofline model; assumptions and sources are in sol.yaml. "
        "Measured overhead is conditional, not a hardware limit. "
        "Region headroom is not additive and is not predicted end-to-end gain. "
        "A measurement above a bound contradicts the model; do not use it as a goal.",
    ]
    return "\n".join(lines)


def describe(sol: SolCeilings | None, baseline: Measurement | None) -> str:
    if sol is None or baseline is None:
        return ""
    return "## Speed of light\n\n" + render_table(sol, [baseline]) + "\n"


def as_state(sol: SolCeilings) -> dict[str, Any]:
    return {
        "method": sol.method,
        "ceilings": {k: vars(c) for k, c in sol.ceilings.items()},
        "regions": sol.regions,
        "unknown": dict(sol.unknown),
    }


def from_state(data: Mapping[str, Any]) -> SolCeilings | None:
    if data.get("method") != METHOD or "ceilings" not in data:
        return None
    return SolCeilings(
        METHOD,
        {k: Ceiling(**v) for k, v in data["ceilings"].items()},
        regions=data.get("regions", []),
        unknown=data.get("unknown", {}),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="Explicit work/capacities YAML or JSON"
    )
    parser.add_argument("--output", type=Path, required=True, help="Reproducible SOL YAML or JSON")
    args = parser.parse_args()
    try:
        result = calculate(yaml.safe_load(args.input.read_text(encoding="utf-8")))
        text = (
            json.dumps(result, indent=2, allow_nan=False)
            if args.output.suffix == ".json"
            else yaml.safe_dump(result, sort_keys=False)
        )
        args.output.write_text(text + "\n", encoding="utf-8")
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.exit(2, f"SOL input error: {exc}\n")


if __name__ == "__main__":
    main()


__all__ = [
    "BOUNDS",
    "Ceiling",
    "SOL_FILENAME",
    "SolCeilings",
    "as_state",
    "calculate",
    "describe",
    "from_state",
    "load_result",
    "load_sol",
    "render_table",
]
