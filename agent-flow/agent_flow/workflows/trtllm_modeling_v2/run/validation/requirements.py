# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What covers each feature of the request: the checks the contract owes for it.

A feature is an assembly unit or a configuration variant. One check of any
kind, at any member, covers it. The request is not proven feature by member:
that product is model loads an onboarding would pay for and learn little from.
A feature no check covers is named at the start and in the delivery report,
and stops nothing.
"""

from __future__ import annotations

from typing import Any, Mapping

from ...spec import variants as variants_mod
from ...spec.task_schema import assembly_units
from . import contract as contract_mod

#: Where the coverage is written, beside ``contract.yaml``.
COVERAGE_FILENAME = "coverage.yaml"


def feature_checks(
    task_data: Mapping[str, Any], contract: contract_mod.Contract
) -> dict[str, list[str]]:
    """Each feature of the request, in its order, and the required checks that cover it."""
    features = [*assembly_units(task_data), *variants_mod.variants(task_data)]
    return {
        feature: [c.key for c in contract.required() if c.unit == feature]
        for feature in dict.fromkeys(features)
    }


def uncovered(coverage: Mapping[str, list[str]]) -> list[str]:
    """The features no check covers."""
    return [feature for feature, checks in coverage.items() if not checks]


def render(coverage: Mapping[str, list[str]]) -> str:
    """A markdown table, one row per feature."""
    lines = ["| feature | covered by |", "|---|---|"]
    for feature, checks in coverage.items():
        shown = ", ".join(f"`{key}`" for key in checks) or "**no check**"
        lines.append(f"| {feature} | {shown} |")
    return "\n".join(lines)


__all__ = [
    "COVERAGE_FILENAME",
    "feature_checks",
    "render",
    "uncovered",
]
