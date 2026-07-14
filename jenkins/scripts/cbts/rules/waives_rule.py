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
"""WaivesRule — narrows CI when waives.txt entries change."""

from __future__ import annotations

from typing import Optional

from blocks import Stage, YAMLIndex, normalize_test_id

from ._helpers import (
    iter_diff_changes,
    lookup_ids_into_block_filters,
    resolve_affected_stages,
    stages_by_yaml_stem,
)
from .base import PRInputs, Rule, RuleResult

WAIVES_FILE = "tests/integration/test_lists/waives.txt"


def _extract_test_id(line: str) -> Optional[str]:
    """Return the normalized test id from a waives.txt line, or None."""
    s = line.strip()
    if not s or s.startswith("#"):
        return None
    s = normalize_test_id(s)
    return s or None


class WaivesRule(Rule):
    name = "waives"
    needs_diff_for = [WAIVES_FILE]

    def __init__(self, yaml_index: YAMLIndex, stages: dict[str, Stage]) -> None:
        self.yaml_index = yaml_index
        self._stages_by_yaml = stages_by_yaml_stem(stages)

    def apply(self, pr: PRInputs) -> Optional[RuleResult]:
        if WAIVES_FILE not in pr.changed_files:
            return None

        added: set[str] = set()
        removed: set[str] = set()
        for sign, body in iter_diff_changes(pr.diffs.get(WAIVES_FILE, "")):
            tid = _extract_test_id(body)
            if tid is None:
                continue
            (added if sign == "+" else removed).add(tid)
        changed_test_ids = added | removed

        if not changed_test_ids:
            # Whitespace / comment-only diff -> no test selection impact.
            return RuleResult(
                handled_files={WAIVES_FILE},
                affected_stages=set(),
                scope="noop",
                sanity_relevant=False,
                perfsanity_relevant=False,
                reason="waives.txt: no actionable test ids in diff",
            )

        # Unmatchable waives (no YAML entry shares lineage) are pre-merge
        # no-ops: the test isn't in any pre-merge YAML test list, so
        # adding/removing its SKIP doesn't affect what runs.
        block_filters, misses = lookup_ids_into_block_filters(self.yaml_index, changed_test_ids)

        if not block_filters:
            # All waives unmatchable -> scope="noop" so other rules can
            # union normally and pure-waives all-miss PRs pass through to
            # Groovy as a no-op rather than tripping Selector's empty-
            # stages safety net.
            preview = ", ".join(sorted(misses)[:3])
            more = f" (+{len(misses) - 3} more)" if len(misses) > 3 else ""
            return RuleResult(
                handled_files={WAIVES_FILE},
                affected_stages=set(),
                scope="noop",
                sanity_relevant=False,
                perfsanity_relevant=False,
                reason=f"waives.txt: all {len(misses)} waive(s) unmatchable: {preview}{more}",
            )

        # Some matched -> normal narrow path; any unmatchable misses are
        # ignored (they don't affect pre-merge) and noted in the reason.
        affected_stages = resolve_affected_stages(
            block_filters, self.yaml_index, self._stages_by_yaml
        )
        sanity_relevant = any(stem == "l0_sanity_check" for stem, _ in block_filters)

        miss_note = ""
        if misses:
            preview = ", ".join(sorted(misses)[:3])
            more = f" (+{len(misses) - 3} more)" if len(misses) > 3 else ""
            miss_note = f"; {len(misses)} unmatchable waive(s) ignored: {preview}{more}"

        return RuleResult(
            handled_files={WAIVES_FILE},
            affected_stages=affected_stages,
            scope="waiveonly",
            block_filters=block_filters,
            sanity_relevant=sanity_relevant,
            perfsanity_relevant=False,
            reason=(
                f"waives.txt: +{len(added)} / -{len(removed)} → "
                f"{len(block_filters)} blocks, {len(affected_stages)} stages{miss_note}"
            ),
        )
