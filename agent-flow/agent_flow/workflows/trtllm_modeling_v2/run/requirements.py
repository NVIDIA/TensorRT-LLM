# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the run claims to support: the request as a matrix, and the checks behind each cell.

The contract (``contract.py``) is a list of checks; it says what will be
measured, not what is claimed. This module turns the request into the claim:
one row per feature the target must serve (each assembly unit, each
configuration variant, and each pair of units whose interaction a unit's own
checks do not reach), one column per member, and per cell the checks that
prove it:

- ``certified``: a check that judges the feature's output runs at that member
  (accuracy, logits, a witness, acceptance);
- ``sampled``: the feature boots at that member and its output is judged at
  others, named in the cell (accuracy is sampled at the members
  ``contract.benchmark_members`` picks);
- ``not covered``: nothing runs there, or nothing judges the feature's output
  anywhere, with the reason.

What judges a feature is its oracle (``checks/observables.py``): for the
``multimodal`` unit, only a check that sends images -- the logit gate over a
reference with image prompts, or an accuracy benchmark that sends images. A
keyword match on the boot's image prompt is not one.

A cell that is not covered is a decision, not an accident: the run stops at
preflight until ``task.yaml`` accepts it (``coverage.accept``, a list of
``<feature>@<member>``, ``<feature>@*`` for a whole row), and the delivery
report carries the matrix, so what "fully supported" means is written down.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from ..checks import logit_agreement, observables
from ..onboard.logit_reference import ReferenceError as LogitReferenceError
from ..onboard.logit_reference import load_reference
from ..spec import variants as variants_mod
from ..spec.task_schema import COVERAGE_FIELD, assembly_units, members_for
from . import contract as contract_mod

#: Where the matrix is written, beside ``contract.yaml``.
COVERAGE_FILENAME = "coverage.yaml"

CERTIFIED, SAMPLED, NOT_COVERED = "certified", "sampled", "not covered"
#: Gates that judge a feature's output, as opposed to proving it starts.
_JUDGING = ("accuracy[", "logits", "acceptance", "replay[", "equivalence", "residency", "kv_quant")
#: Pairs of units whose interaction no unit's own checks reach, and why.
INTERACTIONS: dict[tuple[str, str], str] = {
    ("disagg", "multimodal"): (
        "the disaggregated witness sends text prompts only: image inputs across the "
        "context/generation hand-over are proven by no check"
    ),
}


@dataclass(frozen=True)
class Cell:
    """One feature at one member: how it is covered, by which checks, and why not when not."""

    feature: str
    member: str
    status: str
    checks: tuple[str, ...] = ()
    reason: str = ""
    accepted: bool = False


@dataclass(frozen=True)
class Matrix:
    """The request as feature x member cells."""

    features: tuple[str, ...]
    members: tuple[str, ...]
    cells: tuple[Cell, ...]
    workload: str = ""
    accepted: tuple[str, ...] = field(default_factory=tuple)

    def cell(self, feature: str, member: str) -> Cell:
        return next(c for c in self.cells if c.feature == feature and c.member == member)

    def gaps(self) -> list[Cell]:
        """Cells nothing covers and ``task.yaml`` does not accept."""
        return [c for c in self.cells if c.status == NOT_COVERED and not c.accepted]

    def render(self) -> str:
        """The matrix as a markdown table, then each cell that is not certified, with why."""
        head = "| feature | " + " | ".join(self.members) + " |"
        rule = "|---|" + "---|" * len(self.members)
        mark = {CERTIFIED: "certified", SAMPLED: "sampled", NOT_COVERED: "not covered"}
        rows = []
        for feature in self.features:
            row = []
            for member in self.members:
                cell = self.cell(feature, member)
                text = mark[cell.status]
                if cell.status == NOT_COVERED and cell.accepted:
                    text += " (accepted)"
                row.append(text)
            rows.append(f"| {feature} | " + " | ".join(row) + " |")
        notes = [
            f"- {c.feature}@{c.member}: {c.status}"
            + (" (accepted)" if c.accepted else "")
            + f": {c.reason}"
            for c in self.cells
            if c.status != CERTIFIED
        ]
        workload = f"Workload: {self.workload}.\n\n" if self.workload else ""
        return (
            workload + "\n".join([head, rule, *rows]) + ("\n\n" + "\n".join(notes) if notes else "")
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "features": list(self.features),
            "members": list(self.members),
            "workload": self.workload,
            "accepted": list(self.accepted),
            "cells": [asdict(c) for c in self.cells],
        }


def accepted(task_data: Mapping[str, Any]) -> tuple[str, ...]:
    """The cells ``task.yaml`` accepts as not covered (``coverage.accept``)."""
    block = task_data.get(COVERAGE_FIELD) or {}
    raw = block.get("accept") if isinstance(block, Mapping) else None
    return tuple(str(x) for x in raw) if isinstance(raw, list) else ()


def _accepts(rules: Sequence[str], feature: str, member: str) -> bool:
    return f"{feature}@{member}" in rules or f"{feature}@*" in rules


#: The unit whose output the oracle judges on images only.
MULTIMODAL_UNIT = "multimodal"


def _judging(gate: str) -> bool:
    return any(gate.startswith(prefix) for prefix in _JUDGING)


def _judges(gate: str, feature: str, images: bool) -> bool:
    """Whether a check of ``gate`` judges ``feature``'s output; ``images``: the logit reference has image prompts."""
    if not _judging(gate):
        return False
    if feature != MULTIMODAL_UNIT:
        return True
    if gate == logit_agreement.GATE_NAME:
        return images
    if gate.startswith("accuracy[") and gate.endswith("]"):
        return observables.judges_images(gate[len("accuracy[") : -1])
    return False


def reference_has_images(task_data: Mapping[str, Any]) -> bool:
    """Whether the logit gate's reference holds an image prompt; ``False`` without one."""
    spec = logit_agreement.LogitsSpec.from_task(task_data)
    if not spec.enabled:
        return False
    try:
        reference = load_reference(spec.reference)
    except (LogitReferenceError, OSError):
        return False
    return any(p.images for p in reference.prompts)


def _workload(task_data: Mapping[str, Any]) -> str:
    perf = task_data.get("perf") or {}
    load = perf.get("workload") if isinstance(perf, Mapping) else None
    if isinstance(load, Mapping) and load.get("isl") and load.get("osl"):
        return f"ISL {load['isl']} / OSL {load['osl']}"
    return ""


def requirements_matrix(
    task_data: Mapping[str, Any], contract: contract_mod.Contract, world_size: int
) -> Matrix:
    """The request as a matrix, each cell covered by ``contract``'s checks or not, and why."""
    units = assembly_units(task_data)
    names = list(variants_mod.variants(task_data))
    declared = {u: members_for(task_data, u, world_size) for u in units}
    by_feature: dict[str, dict[str, list[contract_mod.Check]]] = {}
    for check in contract.checks:
        by_feature.setdefault(check.unit, {}).setdefault(check.member, []).append(check)
    members: list[str] = []
    for unit in units:
        members += [m for m in declared[unit] if m not in members]
    for check in contract.checks:
        if check.member not in members:
            members.append(check.member)
    rules = accepted(task_data)
    images = reference_has_images(task_data)
    features: list[str] = [*units, *names]
    cells: list[Cell] = []
    for feature in features:
        at = by_feature.get(feature, {})

        def judges(checks: Sequence[contract_mod.Check], feature: str = feature) -> bool:
            return any(_judges(c.gate, feature, images) for c in checks)

        judged = sorted(m for m, checks in at.items() if judges(checks))
        own = declared.get(feature) or sorted(at)
        for member in members:
            checks = at.get(member, [])
            keys = tuple(c.key for c in checks)
            if judges(checks):
                status, reason = CERTIFIED, ""
            elif checks and judged:
                status = SAMPLED
                reason = f"boots here; its output is judged at {', '.join(judged)}"
            elif checks:
                status = NOT_COVERED
                reason = "boots here, and no check judges its output at any member" + (
                    ": its oracle is "
                    + observables.MULTIMODAL.oracle
                    + " (onboard/logit_reference.py --image, named in accuracy.logits.reference)"
                    if feature == MULTIMODAL_UNIT
                    else ""
                )
            else:
                status = NOT_COVERED
                reason = f"{feature} is gated at {', '.join(own) or 'no member'} only"
            cells.append(
                Cell(feature, member, status, keys, reason, _accepts(rules, feature, member))
            )
    for (first, second), why in INTERACTIONS.items():
        if first in units and second in units:
            pair = f"{first}+{second}"
            features.append(pair)
            for member in members:
                cells.append(
                    Cell(pair, member, NOT_COVERED, (), why, _accepts(rules, pair, member))
                )
    return Matrix(
        features=tuple(features),
        members=tuple(members),
        cells=tuple(cells),
        workload=_workload(task_data),
        accepted=rules,
    )


def gap_message(matrix: Matrix) -> str:
    """Why the run cannot claim full support as specified, and how to accept what it leaves out."""
    lines = [f"- {c.feature}@{c.member}: {c.reason}" for c in matrix.gaps()]
    return (
        "the contract leaves these requested cells uncovered, and task.yaml does not accept "
        "them:\n"
        + "\n".join(lines)
        + "\nCover them (add the members to the feature's `members`, or a check that reaches "
        "them), or accept each as not covered in task.yaml:\n"
        "  coverage:\n    accept: [<feature>@<member>, <feature>@*]\n"
        "What is accepted is written in the delivery report, so the claim matches what was proven."
    )


__all__ = [
    "CERTIFIED",
    "COVERAGE_FILENAME",
    "INTERACTIONS",
    "MULTIMODAL_UNIT",
    "NOT_COVERED",
    "SAMPLED",
    "Cell",
    "Matrix",
    "accepted",
    "gap_message",
    "reference_has_images",
    "requirements_matrix",
]
