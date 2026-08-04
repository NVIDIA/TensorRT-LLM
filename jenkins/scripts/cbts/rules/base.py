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
    """PR data rules query. `post_merge` reflects /bot run --post-merge."""

    changed_files: list[str]
    diffs: dict[str, str]
    post_merge: bool = False


@dataclass
class RuleResult:
    """One rule's contribution.

    `block_filters` is per-block `{filter_prefix: {originating_waive_id, ...}}`
    for Layer 3 narrowing.

    `sanity_relevant` (default True = safe): set False when this rule's
    matched changes have nothing the wheel-sanity check would verify, so
    PackageSanityCheck can be skipped.
    """

    handled_files: set[str]
    affected_stages: set[str]
    scope: Optional[str]
    reason: str
    block_filters: dict[tuple[str, int], dict[str, set[str]]] = field(default_factory=dict)
    sanity_relevant: bool = True
    # True (safe default) iff this rule's matched changes might affect perf
    # benchmarks. Set False when matched changes are pure test infra.
    perfsanity_relevant: bool = True


class Rule(ABC):
    """Base class for CBTS rules.

    `name`: log/reason identifier.
    `needs_diff_for`: file paths/globs whose diffs the rule consumes.
    `apply(pr)`: return RuleResult or None (not applicable).
    """

    name: str = ""
    needs_diff_for: tuple[str, ...] = ()

    @abstractmethod
    def apply(self, pr: PRInputs) -> Optional[RuleResult]: ...
