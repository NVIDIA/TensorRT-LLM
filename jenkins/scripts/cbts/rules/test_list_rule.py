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
"""TestListRule — narrows CI when test-db YAML files change.

Only ADDED entries drive narrowing. Removed entries don't trigger
verification: the test either still runs elsewhere (unchanged) or is
fully retired. Comment lines and whitespace-only edits are ignored;
any other non-entry +/- line (condition / mako / whole-block change)
falls back since Layer 2 stage matching depends on those.
"""

from __future__ import annotations

import re
from typing import Optional

from blocks import Stage, YAMLIndex, normalize_test_id

from ._helpers import (
    iter_diff_changes,
    lookup_ids_into_block_filters,
    resolve_affected_stages,
    stages_by_yaml_stem,
)
from .base import PRInputs, Rule, RuleResult

TEST_DB_DIR = "tests/integration/test_lists/test-db/"

# `<2+ space indent>- <pytest_id>`. Top-level `- condition:` (column-0
# block-list item) intentionally does not match — the leading `\s+`
# discriminates entry-within-block from block-boundary edits.
_ENTRY_RE = re.compile(r"^\s+-\s+(\S.*)$")
_COMMENT_RE = re.compile(r"^\s*#")


def _is_test_db_file(path: str) -> bool:
    return path.startswith(TEST_DB_DIR) and path.endswith(".yml")


def _is_perf_stem(stem: str) -> bool:
    return stem == "l0_perf" or "perf_sanity" in stem


class TestListRule(Rule):
    name = "testlist"
    needs_diff_for: tuple[str, ...] = ("tests/integration/test_lists/test-db/*.yml",)

    def __init__(self, yaml_index: YAMLIndex, stages: dict[str, Stage]) -> None:
        self.yaml_index = yaml_index
        self._stages_by_yaml = stages_by_yaml_stem(stages)

    def apply(self, pr: PRInputs) -> Optional[RuleResult]:
        touched = sorted(p for p in pr.changed_files if _is_test_db_file(p))
        if not touched:
            return None

        added: set[str] = set()
        removed: set[str] = set()
        structural: list[str] = []

        for path in touched:
            for sign, body in iter_diff_changes(pr.diffs.get(path, "")):
                if not body.strip():
                    continue
                if _COMMENT_RE.match(body):
                    continue
                m = _ENTRY_RE.match(body)
                if m is None:
                    structural.append(f"{path}:{sign}{body.rstrip()}")
                    continue
                tid = normalize_test_id(m.group(1))
                if tid:
                    (added if sign == "+" else removed).add(tid)

        if structural:
            preview = "; ".join(structural[:3])
            more = f" (+{len(structural) - 3} more)" if len(structural) > 3 else ""
            return RuleResult(
                handled_files=set(touched),
                affected_stages=set(),
                scope=None,
                reason=f"testlist: {len(structural)} structural change(s): {preview}{more}",
            )

        if not added:
            # Only removals (or comment/whitespace-only edits) → no new
            # tests to verify; claim as noop so other rules union normally.
            note = f" ({len(removed)} removed entries don't need verification)" if removed else ""
            return RuleResult(
                handled_files=set(touched),
                affected_stages=set(),
                scope="noop",
                sanity_relevant=False,
                perfsanity_relevant=False,
                reason=f"testlist: no additions across {len(touched)} file(s){note}",
            )

        block_filters, misses = lookup_ids_into_block_filters(self.yaml_index, added)
        if not block_filters:
            # All added entries unresolvable → no narrow contribution; claim
            # as noop (analogous to waivesonly all-miss).
            preview = ", ".join(sorted(misses)[:3])
            more = f" (+{len(misses) - 3} more)" if len(misses) > 3 else ""
            return RuleResult(
                handled_files=set(touched),
                affected_stages=set(),
                scope="noop",
                sanity_relevant=False,
                perfsanity_relevant=False,
                reason=(
                    f"testlist: all {len(misses)} added entry/entries don't resolve "
                    f"to any post-PR YAML block: {preview}{more}"
                ),
            )

        affected_stages = resolve_affected_stages(
            block_filters, self.yaml_index, self._stages_by_yaml
        )
        sanity_relevant = any(stem == "l0_sanity_check" for stem, _ in block_filters)
        perfsanity_relevant = any(_is_perf_stem(stem) for stem, _ in block_filters)

        miss_note = ""
        if misses:
            preview = ", ".join(sorted(misses)[:3])
            more = f" (+{len(misses) - 3} more)" if len(misses) > 3 else ""
            miss_note = f"; {len(misses)} unresolved entry/entries ignored: {preview}{more}"

        return RuleResult(
            handled_files=set(touched),
            affected_stages=affected_stages,
            scope="testlistonly",
            block_filters=block_filters,
            sanity_relevant=sanity_relevant,
            perfsanity_relevant=perfsanity_relevant,
            reason=(
                f"testlist: +{len(added)} / -{len(removed)} across "
                f"{len(touched)} file(s) → narrow on +{len(added)} → "
                f"{len(block_filters)} blocks, {len(affected_stages)} stages"
                f"{miss_note}"
            ),
        )
