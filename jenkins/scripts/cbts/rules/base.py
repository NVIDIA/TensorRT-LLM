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
    """Inputs about the PR that rules can query.

    `post_merge` reflects the user's `/bot run [--post-merge]` flag. Rules
    themselves don't consult it (their narrowing is mode-agnostic); main.py
    uses it after `Selector.run` to drop `affected_stages` entries that
    don't match the trigger mode (pre-merge vs Post-Merge by stage-name
    convention). See `main.py::main` for the filter; default False keeps
    backward compat with older Groovy that didn't pass the field.
    """

    changed_files: list[str]
    diffs: dict[str, str]
    post_merge: bool = False


@dataclass
class RuleResult:
    """What a single rule contributes when it applies to a PR.

    `block_filters` (CBTS Layer 3): per-block map of filter prefix -> set of
    waive ids that resolved to that prefix. Each affected block (keyed by
    `(yaml_stem, block_index)`) maps to {prefix: {waive_id, ...}}. The
    Selector aggregates this across rules and `write_filtered_test_db`
    uses both the prefix (subtree match) AND the waive ids (to skip YAML
    entries whose `-k "<keyword>"` filter doesn't match the waived test).
    Empty when the rule doesn't produce Layer 3 narrowing.
    """

    handled_files: set[str]
    tests: set[str]
    affected_stages: set[str]
    scope: Optional[str]
    reason: str
    block_filters: dict[tuple[str, int], dict[str, set[str]]] = field(default_factory=dict)


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
