# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The runtime map: which engine area a checkpoint needs, and where it lives.

The map carries only where the code is and what to grep for, and a sentence or
two saying what the area *is*: no signatures, no argument lists, no line
numbers, no copied code. :func:`areas_for_demands` takes the structural demands
``survey.py`` read off ``config.json`` and returns the areas they imply.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import yaml

#: Ships with the package; see ``pyproject.toml``'s package-data.
INDEX_PATH = Path(__file__).parent / "runtime_index.yaml"
#: Where a run keeps the map, narrowed to its checkpoint, in its workspace.
RUNTIME_MAP_FILENAME = "runtime_map.md"

KERNEL_SOURCES_PATH = Path(__file__).parent / "kernel_sources.yaml"

KERNEL_SOURCE_REQUIRED: tuple[str, ...] = ("summary", "tier", "availability", "certifying")
KERNEL_SOURCE_OPTIONAL: tuple[str, ...] = (
    "registers",
    "implementation",
    "claims_files_containing",
    "package",
)

#: Areas every run needs regardless of what the checkpoint declares.
ALWAYS_RELEVANT: tuple[str, ...] = (
    "parallelism",
    "cuda_graph",
    "scheduling",
    "kv_cache",
    "jit_autotune",
)

REQUIRED_FIELDS: tuple[str, ...] = ("summary", "code", "gating")
OPTIONAL_FIELDS: tuple[str, ...] = ("symbols", "tests", "doc", "demands")

GATING_FIELDS: tuple[str, ...] = ("gate", "signal", "blind")

#: Which existing gate covers an area. ``none`` is the flag that says a
#: witness has to be built before the feature can be claimed.
GATES: tuple[str, ...] = ("boot", "accuracy", "acceptance", "none")

#: The one non-area key. It holds the shape of the engine rather than one
#: part of it, so it has its own schema: a summary and an ordered `flow` of
#: stages, each naming the areas that live there.
OVERVIEW_KEY = "_overview"
FLOW_FIELDS: tuple[str, ...] = ("stage", "does", "areas")


class RuntimeIndexError(ValueError):
    """Raised when the index is malformed or points somewhere that is not there."""


@dataclass(frozen=True)
class Gating:
    """How a run proves this area works, and what it cannot see."""

    gate: str
    signal: str
    blind: str


@dataclass(frozen=True)
class RuntimeArea:
    """One engine area, and where to read it."""

    name: str
    summary: str
    gating: "Gating | None" = None
    code: tuple[str, ...] = ()
    symbols: tuple[str, ...] = ()
    tests: tuple[str, ...] = ()
    doc: str | None = None
    demands: tuple[str, ...] = ()

    def render(self) -> str:
        """The block an agent reads. Pointers, not content."""
        lines = [f"### {self.name}", "", " ".join(self.summary.split()), ""]
        lines.append(f"- **code**: {', '.join(f'`{p}`' for p in self.code)}")
        if self.symbols:
            lines.append(f"- **grep for**: {', '.join(f'`{s}`' for s in self.symbols)}")
        if self.tests:
            lines.append(f"- **tests**: {', '.join(f'`{p}`' for p in self.tests)}")
        if self.doc:
            lines.append(f"- **guide**: `{self.doc}`")
        if self.gating:
            lines.append(f"- **proven by**: {self.gating.gate} gate — {self.gating.signal}")
            lines.append(f"- **blind spot**: {self.gating.blind}")
        return "\n".join(lines)


@dataclass(frozen=True)
class FlowStage:
    """One stage of the request path, and which areas live there."""

    stage: str
    does: str
    areas: tuple[str, ...] = ()


@dataclass(frozen=True)
class Overview:
    """The engine's shape: what a target is one link of."""

    summary: str = ""
    flow: tuple[FlowStage, ...] = ()

    def render(self) -> str:
        lines = [
            "## How the engine fits together",
            "",
            " ".join(self.summary.split()),
            "",
            "| stage | what it owns | areas |",
            "|---|---|---|",
        ]
        for stage in self.flow:
            areas = ", ".join(f"`{a}`" for a in stage.areas) or "—"
            lines.append(f"| **{stage.stage}** | {' '.join(stage.does.split())} | {areas} |")
        return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class RuntimeIndex:
    """The whole map, loaded once."""

    areas: dict[str, RuntimeArea] = field(default_factory=dict)
    overview: Overview = field(default_factory=Overview)

    def __iter__(self):
        return iter(self.areas.values())

    def by_demand(self, demand: str) -> tuple[RuntimeArea, ...]:
        """Areas that declare ``demand`` as something that makes them relevant."""
        return tuple(a for a in self.areas.values() if demand in a.demands)


@dataclass(frozen=True)
class KernelSource:
    """One place kernels come from, and what it costs to certify one."""

    name: str
    summary: str
    tier: int
    availability: str
    certifying: str
    registers: tuple[str, ...] = ()
    implementation: tuple[str, ...] = ()
    #: Markers that claim a `custom_ops/*.py` file `registers` does not name:
    #: a file containing every one of them belongs to this source.
    claims_files_containing: tuple[str, ...] = ()
    #: The pip distribution the kernels ship in, for a source that is a
    #: dependency rather than tree code; it is pinned in ``requirements.txt``.
    package: str = ""

    def render(self) -> str:
        lines = [f"### {self.name} (tier {self.tier})", "", " ".join(self.summary.split()), ""]
        if self.registers:
            lines.append(f"- **registers through**: {', '.join(f'`{p}`' for p in self.registers)}")
        if self.implementation:
            lines.append(
                f"- **implemented in**: {', '.join(f'`{p}`' for p in self.implementation)}"
            )
        if self.claims_files_containing:
            markers = " and ".join(f"`{m}`" for m in self.claims_files_containing)
            lines.append(f"- **also any `custom_ops/*.py` containing**: {markers}")
        if self.package:
            lines.append(f"- **package**: `{self.package}`, pinned in `requirements.txt`")
        lines.append(f"- **availability**: {' '.join(self.availability.split())}")
        lines.append(f"- **certifying one**: {' '.join(self.certifying.split())}")
        return "\n".join(lines)


def load_kernel_sources(path: str | Path = KERNEL_SOURCES_PATH) -> dict[str, KernelSource]:
    """Parse and validate the kernel-source registry."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not raw:
        raise RuntimeIndexError(f"{path} must be a non-empty mapping of source -> entry")

    sources: dict[str, KernelSource] = {}
    errors: list[str] = []
    for name, entry in raw.items():
        if not isinstance(entry, dict):
            errors.append(f"{name}: must be a mapping")
            continue
        unknown = sorted(set(entry) - set(KERNEL_SOURCE_REQUIRED) - set(KERNEL_SOURCE_OPTIONAL))
        if unknown:
            errors.append(f"{name}: unknown field(s) {unknown}")
        for required in KERNEL_SOURCE_REQUIRED:
            if entry.get(required) in (None, "", []):
                errors.append(f"{name}: missing required field '{required}'")
        if not any(entry.get(key) for key in ("registers", "implementation", "package")):
            errors.append(
                f"{name}: needs at least one of 'registers', 'implementation' or 'package'"
            )
        markers = entry.get("claims_files_containing", [])
        if not isinstance(markers, list) or not all(isinstance(m, str) and m for m in markers):
            errors.append(f"{name}: 'claims_files_containing' must be a list of non-empty strings")
            markers = []
        sources[str(name)] = KernelSource(
            name=str(name),
            summary=str(entry.get("summary", "")),
            tier=int(entry.get("tier", 0) or 0),
            availability=str(entry.get("availability", "")),
            certifying=str(entry.get("certifying", "")),
            registers=tuple(str(p) for p in entry.get("registers", ())),
            implementation=tuple(str(p) for p in entry.get("implementation", ())),
            claims_files_containing=tuple(markers),
            package=str(entry.get("package", "") or ""),
        )
    if errors:
        raise RuntimeIndexError(f"{path} is malformed:\n  - " + "\n  - ".join(errors))
    return sources


def format_kernel_sources(sources: "Mapping[str, KernelSource]") -> str:
    """The block a kernel writer reads before deciding where a computation comes from."""
    lines = [
        "## Where kernels come from",
        "",
        "An op's presence is measured on the host --- the surface survey above "
        "enumerated what this build actually registers. This says what each "
        "source IS and what certifying one of its kernels requires, because "
        "that differs per source and decides the shape of your receipt.",
        "",
        "Both registration planes land in the same `torch.ops.trtllm` namespace, "
        "so the namespace alone tells you nothing about provenance. An op found "
        "in no `custom_ops/*.py` file is almost certainly registered from C++.",
        "",
        "Not every kernel the engine ships is a `torch.ops.trtllm` op: FlashInfer's "
        "public API (`flashinfer_api` below) holds kernels no trtllm op wraps, and "
        "the survey lists them separately.",
        "",
    ]
    lines.extend(source.render() + "\n" for source in sources.values())
    return "\n".join(lines).rstrip() + "\n"


def load_index(path: str | Path = INDEX_PATH) -> RuntimeIndex:
    """Parse and validate the index."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not raw:
        raise RuntimeIndexError(f"{path} must be a non-empty mapping of area -> entry")

    areas: dict[str, RuntimeArea] = {}
    errors: list[str] = []

    overview = Overview()
    raw_overview = raw.pop(OVERVIEW_KEY, None)
    if raw_overview is not None:
        if not isinstance(raw_overview, dict):
            errors.append(f"{OVERVIEW_KEY}: must be a mapping")
        else:
            stages = []
            for item in raw_overview.get("flow", ()):
                if not isinstance(item, dict):
                    errors.append(f"{OVERVIEW_KEY}.flow: each stage must be a mapping")
                    continue
                unknown_flow = sorted(set(item) - set(FLOW_FIELDS))
                if unknown_flow:
                    errors.append(f"{OVERVIEW_KEY}.flow: unknown field(s) {unknown_flow}")
                stages.append(
                    FlowStage(
                        stage=str(item.get("stage", "")),
                        does=str(item.get("does", "")),
                        areas=tuple(str(a) for a in item.get("areas", ())),
                    )
                )
            overview = Overview(summary=str(raw_overview.get("summary", "")), flow=tuple(stages))

    for name, entry in raw.items():
        if str(name).startswith("_"):
            errors.append(f"{name}: unknown reserved key; only {OVERVIEW_KEY!r} is reserved")
            continue
        if not isinstance(entry, dict):
            errors.append(f"{name}: must be a mapping, got {type(entry).__name__}")
            continue
        unknown = sorted(set(entry) - set(REQUIRED_FIELDS) - set(OPTIONAL_FIELDS))
        if unknown:
            errors.append(f"{name}: unknown field(s) {unknown}")
        for required in REQUIRED_FIELDS:
            if not entry.get(required):
                errors.append(f"{name}: missing required field '{required}'")

        gating = None
        raw_gating = entry.get("gating")
        if isinstance(raw_gating, dict):
            unknown_gating = sorted(set(raw_gating) - set(GATING_FIELDS))
            if unknown_gating:
                errors.append(f"{name}.gating: unknown field(s) {unknown_gating}")
            for required in GATING_FIELDS:
                if not raw_gating.get(required):
                    errors.append(f"{name}.gating: missing '{required}'")
            gate = str(raw_gating.get("gate", ""))
            if gate and gate not in GATES:
                errors.append(f"{name}.gating.gate is {gate!r}; must be one of {list(GATES)}")
            gating = Gating(
                gate=gate,
                signal=" ".join(str(raw_gating.get("signal", "")).split()),
                blind=" ".join(str(raw_gating.get("blind", "")).split()),
            )
        elif raw_gating is not None:
            errors.append(f"{name}.gating: must be a mapping, got {type(raw_gating).__name__}")
        if errors and name not in {e.split(":")[0] for e in errors}:
            continue
        areas[str(name)] = RuntimeArea(
            name=str(name),
            summary=str(entry.get("summary", "")),
            code=tuple(str(p) for p in entry.get("code", ())),
            symbols=tuple(str(s) for s in entry.get("symbols", ())),
            tests=tuple(str(p) for p in entry.get("tests", ())),
            doc=str(entry["doc"]) if entry.get("doc") else None,
            demands=tuple(str(d) for d in entry.get("demands", ())),
            gating=gating,
        )
    if errors:
        raise RuntimeIndexError(f"{path} is malformed:\n  - " + "\n  - ".join(errors))
    return RuntimeIndex(areas=areas, overview=overview)


def areas_for_demands(demands: Iterable[str], *, index: RuntimeIndex) -> tuple[RuntimeArea, ...]:
    """Return the areas the checkpoint's measured demands imply.

    Returned in index order rather than match order, so two runs on the same
    checkpoint read the same document.
    """
    wanted = set(ALWAYS_RELEVANT)
    for demand in demands:
        wanted.update(a.name for a in index.by_demand(demand))
    return tuple(a for a in index if a.name in wanted)


def format_areas(
    areas: Sequence[RuntimeArea],
    *,
    unmatched: Sequence[str] = (),
    overview: "Overview | None" = None,
) -> str:
    """Render the map block for a prompt."""
    if not areas:
        return ""
    lines = []
    if overview is not None and overview.flow:
        lines += [overview.render(), ""]
    lines += [
        "## Runtime map",
        "",
        "The engine areas this checkpoint touches, and where each one lives **in "
        "this checkout right now**. Read the source; do not take a description of "
        "it from anywhere, including from here. These entries say where to look "
        "and what to grep for, and deliberately carry no signatures, no argument "
        "lists and no line numbers — those go stale between one onboarding and "
        "the next, and a stale one reads exactly like a true one.",
        "",
        "The `tests` path matters as much as the `code` path: a new entry's test "
        "should look like the tests already sitting there.",
        "",
    ]
    lines.extend(area.render() + "\n" for area in areas)
    if unmatched:
        lines.append(
            "**Demands no area claims**: "
            + ", ".join(f"`{d}`" for d in unmatched)
            + ". That is the signal worth acting on — either the map has a gap, "
            "or this checkpoint needs something the engine has no home for. Say "
            "which in the plan."
        )
    return "\n".join(lines).rstrip() + "\n"


def unmatched_demands(demands: Iterable[str], *, index: RuntimeIndex) -> tuple[str, ...]:
    """Demands the map has no area for — a gap in the map or in the engine."""
    return tuple(sorted({d for d in demands if not index.by_demand(d)}))


__all__ = [
    "ALWAYS_RELEVANT",
    "FLOW_FIELDS",
    "FlowStage",
    "GATES",
    "GATING_FIELDS",
    "Gating",
    "INDEX_PATH",
    "KERNEL_SOURCES_PATH",
    "KernelSource",
    "RUNTIME_MAP_FILENAME",
    "RuntimeArea",
    "RuntimeIndex",
    "OVERVIEW_KEY",
    "Overview",
    "RuntimeIndexError",
    "areas_for_demands",
    "format_areas",
    "format_kernel_sources",
    "load_index",
    "load_kernel_sources",
    "unmatched_demands",
]
