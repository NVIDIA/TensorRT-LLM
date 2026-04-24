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
"""WaivesRule — v0 rule for changes to waives.txt."""

from __future__ import annotations

import re
from typing import Optional

from blocks import Stage, YAMLIndex, block_matches_stage

from .base import PRInputs, Rule, RuleResult

WAIVES_FILE = "tests/integration/test_lists/waives.txt"

# Strip GPU/platform prefixes like "full:GH200/" or "full:sm100/" at the start.
_PREFIX_RE = re.compile(r"^full:[^/]+/")

# Split trailing annotations (SKIP / TIMEOUT / comments) from the test id.
# A waive line typically looks like:
#   <test_id> SKIP (reason)
#   <test_id> SKIP # url
#   <test_id> # just a comment
#   <test_id> -k "expr" SKIP (reason)
# We keep the whole "<test_id> [-m/-k ...]" portion as the identifier, since
# YAML entries can include the same -m/-k suffixes.
_SUFFIX_RE = re.compile(r"\s+(SKIP|TIMEOUT)\b.*$")


def _extract_test_id(line: str) -> Optional[str]:
    """Extract the test identifier from a waives.txt line.

    Returns None if the line doesn't look like a waive entry (empty, pure
    comment, etc).
    """
    s = line.strip()
    if not s or s.startswith("#"):
        return None
    # Drop the "SKIP ..." / "TIMEOUT ..." trailing annotation if present.
    s = _SUFFIX_RE.sub("", s).strip()
    # Drop trailing " # comment" if any.
    if "#" in s:
        s = s.split("#", 1)[0].strip()
    if not s:
        return None
    return s


def _strip_prefix(test_id: str) -> str:
    """Strip leading "full:<gpu>/" prefix if any."""
    return _PREFIX_RE.sub("", test_id)


def parse_waives_diff(diff: str) -> tuple[set[str], set[str]]:
    """Parse a unified diff of waives.txt.

    Returns (added, removed) sets of test identifiers, with "full:..." prefixes
    stripped so they can match YAML entries directly.
    """
    added: set[str] = set()
    removed: set[str] = set()
    for line in diff.splitlines():
        if not line or line.startswith(("+++", "---", "@@")):
            continue
        sign, body = line[0], line[1:]
        if sign not in ("+", "-"):
            continue
        tid = _extract_test_id(body)
        if tid is None:
            continue
        tid = _strip_prefix(tid)
        (added if sign == "+" else removed).add(tid)
    return added, removed


class WaivesRule(Rule):
    name = "waives"
    needs_diff_for = [WAIVES_FILE]

    def __init__(self, yaml_index: YAMLIndex, stages: dict[str, Stage]) -> None:
        self.yaml_index = yaml_index
        # Group stages by YAML stem so block->stage lookup is O(stages_in_yaml)
        # instead of O(total_stages) per block.
        self._stages_by_yaml: dict[str, list[tuple[str, Stage]]] = {}
        for name, stage in stages.items():
            self._stages_by_yaml.setdefault(stage.yaml_stem, []).append((name, stage))

    def apply(self, pr: PRInputs) -> Optional[RuleResult]:
        if WAIVES_FILE not in pr.changed_files:
            return None

        diff = pr.diffs.get(WAIVES_FILE, "")
        added, removed = parse_waives_diff(diff)
        changed_test_ids = added | removed
        if not changed_test_ids:
            return RuleResult(
                # TESTING ONLY (revert to {WAIVES_FILE} before merge): claim all
                # changed files so CBTS fires on the cbts-v0 PR that also edits
                # CBTS infra files.
                handled_files=set(pr.changed_files),
                tests=set(),
                affected_stages=set(),
                scope="waiveonly",
                reason="waives.txt: no actionable test ids in diff",
            )

        seen_block_keys: set[tuple[str, int]] = set()
        affected_blocks = []
        for tid in changed_test_ids:
            for block in self.yaml_index.blocks_containing_test(tid):
                key = (block.yaml_stem, block.block_index)
                if key not in seen_block_keys:
                    seen_block_keys.add(key)
                    affected_blocks.append(block)

        affected_stage_names: set[str] = set()
        for block in affected_blocks:
            for stage_name, stage in self._stages_by_yaml.get(block.yaml_stem, []):
                if block_matches_stage(block, stage):
                    affected_stage_names.add(stage_name)

        return RuleResult(
            # TESTING ONLY (revert to {WAIVES_FILE} before merge): claim all
            # changed files so CBTS fires on the cbts-v0 PR that also edits
            # CBTS infra files.
            handled_files=set(pr.changed_files),
            tests=changed_test_ids,
            affected_stages=affected_stage_names,
            scope="waiveonly",
            reason=(
                f"waives.txt: +{len(added)} / -{len(removed)} → "
                f"{len(affected_blocks)} blocks, {len(affected_stage_names)} stages"
            ),
        )
