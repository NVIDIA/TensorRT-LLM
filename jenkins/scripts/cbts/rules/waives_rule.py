# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
        self.stages = stages

    def apply(self, pr: PRInputs) -> Optional[RuleResult]:
        if WAIVES_FILE not in pr.changed_files:
            return None

        diff = pr.diffs.get(WAIVES_FILE, "")
        added, removed = parse_waives_diff(diff)
        changed_test_ids = added | removed
        if not changed_test_ids:
            # PR touched waives.txt but diff has no parseable test ids (e.g.
            # a pure comment edit). Still claim handling — no stages needed.
            return RuleResult(
                handled_files={WAIVES_FILE},
                tests=set(),
                affected_stages=set(),
                scope="waiveonly",
                reason="waives.txt: no actionable test ids in diff",
            )

        # Reverse-lookup: test id -> containing blocks, deduped.
        seen_block_keys: set[tuple[str, int]] = set()
        affected_blocks = []
        for tid in changed_test_ids:
            for block in self.yaml_index.blocks_containing_test(tid):
                key = (block.yaml_stem, block.block_index)
                if key not in seen_block_keys:
                    seen_block_keys.add(key)
                    affected_blocks.append(block)

        # For each block, find stages whose mako matches its condition.
        affected_stage_names: set[str] = set()
        for block in affected_blocks:
            for stage_name, stage in self.stages.items():
                if stage.yaml_stem != block.yaml_stem:
                    continue
                if block_matches_stage(block, stage):
                    affected_stage_names.add(stage_name)

        return RuleResult(
            handled_files={WAIVES_FILE},
            tests=changed_test_ids,
            affected_stages=affected_stage_names,
            scope="waiveonly",
            reason=(
                f"waives.txt: +{len(added)} / -{len(removed)} → "
                f"{len(affected_blocks)} blocks, {len(affected_stage_names)} stages"
            ),
        )
