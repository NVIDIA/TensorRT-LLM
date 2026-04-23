# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rule contract and shared data types for CBTS."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PRInputs:
    """Inputs about the PR that rules can query."""

    changed_files: list[str]
    diffs: dict[str, str] = field(default_factory=dict)


@dataclass
class RuleResult:
    """What a single rule contributes when it applies to a PR."""

    handled_files: set[str]
    tests: set[str]
    affected_stages: set[str]
    scope: str
    reason: str


class Rule(ABC):
    """Base class for all CBTS rules.

    A rule declares:
      - `name`: identifier used in logs/reasons
      - `needs_diff_for`: file paths / glob patterns whose diffs this rule consumes
                         (Groovy uses this to decide which files to fetch diffs for)

    Subclasses implement `apply(pr)` returning either None (not applicable) or
    a RuleResult.
    """

    name: str = ""
    needs_diff_for: list[str] = []

    @abstractmethod
    def apply(self, pr: PRInputs) -> Optional[RuleResult]: ...
