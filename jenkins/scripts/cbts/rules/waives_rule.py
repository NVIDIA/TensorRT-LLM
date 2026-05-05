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

from typing import Optional

from blocks import Block, Stage, YAMLIndex, block_matches_stage, normalize_test_id

from .base import PRInputs, Rule, RuleResult

WAIVES_FILE = "tests/integration/test_lists/waives.txt"


def _extract_test_id(line: str) -> Optional[str]:
    """Return the normalized test id from a waives.txt line, or None."""
    s = line.strip()
    if not s or s.startswith("#"):
        return None
    s = normalize_test_id(s)
    return s or None


def parse_waives_diff(diff: str) -> tuple[set[str], set[str]]:
    """Return (added, removed) normalized test ids from a waives.txt diff."""
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
        (added if sign == "+" else removed).add(tid)
    return added, removed


class WaivesRule(Rule):
    name = "waives"
    needs_diff_for = [WAIVES_FILE]

    def __init__(self, yaml_index: YAMLIndex, stages: dict[str, Stage]) -> None:
        self.yaml_index = yaml_index
        # Group stages by YAML stem for O(stages_in_yaml) per-block lookup.
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
                handled_files={WAIVES_FILE},
                affected_stages=set(),
                scope="waiveonly",
                reason="waives.txt: no actionable test ids in diff",
            )

        # Walk each waive's parent chain to the first matching YAML level;
        # any miss → scope=None (fallback). Record per-block prefix →
        # {originating waive ids} for write_filtered_test_db's `-k` re-check.
        block_filters: dict[tuple[str, int], dict[str, set[str]]] = {}
        affected_blocks: list[Block] = []
        seen_block_keys: set[tuple[str, int]] = set()
        misses: list[str] = []

        for tid in changed_test_ids:
            match = self.yaml_index.find_match_for_waive(tid)
            if match is None:
                misses.append(tid)
                continue
            level, blocks = match
            for block in blocks:
                key = (block.yaml_stem, block.block_index)
                block_filters.setdefault(key, {}).setdefault(level, set()).add(tid)
                if key not in seen_block_keys:
                    seen_block_keys.add(key)
                    affected_blocks.append(block)

        if misses:
            preview = ", ".join(sorted(misses)[:3])
            more = f" (+{len(misses) - 3} more)" if len(misses) > 3 else ""
            return RuleResult(
                handled_files={WAIVES_FILE},
                affected_stages=set(),
                scope=None,  # Selector treats this as "no decision" → fallback
                reason=f"waives.txt: {len(misses)} unmatchable waive(s): {preview}{more}",
            )

        affected_stage_names: set[str] = set()
        for block in affected_blocks:
            for stage_name, stage in self._stages_by_yaml.get(block.yaml_stem, []):
                if block_matches_stage(block, stage):
                    affected_stage_names.add(stage_name)

        return RuleResult(
            handled_files={WAIVES_FILE},
            affected_stages=affected_stage_names,
            scope="waiveonly",
            block_filters=block_filters,
            sanity_relevant=any(b.yaml_stem == "l0_sanity_check" for b in affected_blocks),
            perfsanity_relevant=False,
            reason=(
                f"waives.txt: +{len(added)} / -{len(removed)} → "
                f"{len(affected_blocks)} blocks, {len(affected_stage_names)} stages"
            ),
        )
