# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Rule contract and shared data types for CBTS."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PRInputs:
    """Inputs about the PR that rules can query."""

    changed_files: list[str]
    diffs: dict[str, str]


@dataclass
class RuleResult:
    """What a single rule contributes when it applies to a PR.

    `block_filters` (CBTS Layer 3): per-block set of filter prefixes. Each
    affected block (keyed by `(yaml_stem, block_index)`) maps to the filter
    levels at which its waive(s) hit. The Selector aggregates this across
    rules and uses it to write a tmp test-db with each affected block's
    `tests:` array narrowed to entries in any filter prefix's subtree.
    Empty when the rule doesn't produce Layer 3 narrowing.
    """

    handled_files: set[str]
    tests: set[str]
    affected_stages: set[str]
    scope: Optional[str]
    reason: str
    block_filters: dict[tuple[str, int], set[str]] = field(default_factory=dict)


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
    needs_diff_for: tuple[str, ...] = ()

    @abstractmethod
    def apply(self, pr: PRInputs) -> Optional[RuleResult]: ...
