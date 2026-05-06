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
"""Pure-function helpers shared by rules in this package."""

from __future__ import annotations

import re
from typing import Iterable, Iterator

from blocks import Stage, YAMLIndex, _entry_target, _target_in_filter_subtree, block_matches_stage

_HUNK_HEADER_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def stages_by_yaml_stem(stages: dict[str, Stage]) -> dict[str, list[tuple[str, Stage]]]:
    """Bucket {stage_name: Stage} by yaml_stem for O(stages_in_yaml) lookup."""
    out: dict[str, list[tuple[str, Stage]]] = {}
    for stage_name, stage in stages.items():
        out.setdefault(stage.yaml_stem, []).append((stage_name, stage))
    return out


def iter_diff_changes(diff: str) -> Iterator[tuple[str, str]]:
    """Yield (sign, body) for every `+`/`-` line; skip headers and context."""
    for line in diff.splitlines():
        if not line or line.startswith(("+++", "---", "@@")):
            continue
        sign = line[0]
        if sign not in ("+", "-"):
            continue
        yield sign, line[1:]


def iter_diff_post_line_numbers(diff: str) -> set[int]:
    """Post-PR line numbers (1-indexed) touched by `+` or `-` lines.

    `+` lines mark their own line; `-` lines anchor at the next post-PR
    line. Hunk headers reset the cursor; context lines advance it.
    """
    out: set[int] = set()
    new_line = 0
    for line in diff.splitlines():
        m = _HUNK_HEADER_RE.match(line)
        if m is not None:
            new_line = int(m.group(1))
            continue
        if not line or line.startswith(("+++", "---")):
            continue
        sign = line[0]
        if sign == "+":
            out.add(new_line)
            new_line += 1
        elif sign == "-":
            out.add(new_line)
        else:
            new_line += 1
    return out


def lookup_ids_into_block_filters(
    yaml_index: YAMLIndex,
    test_ids: Iterable[str],
) -> tuple[dict[tuple[str, int], dict[str, set[str]]], list[str]]:
    """Resolve test ids via `find_match_for_waive`; return (block_filters, misses).

    Each id is added to its matched level's identifier set per affected
    block, so `write_filtered_test_db`'s `-k` keyword guard re-checks
    each entry against the original test ids.
    """
    block_filters: dict[tuple[str, int], dict[str, set[str]]] = {}
    misses: list[str] = []
    for tid in test_ids:
        match = yaml_index.find_match_for_waive(tid)
        if match is None:
            misses.append(tid)
            continue
        level, blocks = match
        for block in blocks:
            key = (block.yaml_stem, block.block_index)
            block_filters.setdefault(key, {}).setdefault(level, set()).add(tid)
    return block_filters, misses


def lookup_paths_into_block_filters(
    yaml_index: YAMLIndex,
    paths: Iterable[str],
) -> tuple[dict[tuple[str, int], dict[str, set[str]]], list[str]]:
    """Resolve YAML-namespace paths via `find_match_for_path`; register each
    covering entry's raw text as identifier so it passes its own `-k` guard.

    Returns (block_filters, unmatched_paths). A path is "unmatched" when
    no YAML entry shares pytest-tree lineage with it. Over-includes
    sibling `-k` variants on purpose: a path change should run every
    parameterization of the affected file/scope.
    """
    block_filters: dict[tuple[str, int], dict[str, set[str]]] = {}
    unmatched: list[str] = []
    for path in paths:
        matches = yaml_index.find_match_for_path(path)
        if matches is None:
            unmatched.append(path)
            continue
        for block, prefix in matches:
            key = (block.yaml_stem, block.block_index)
            ids = block_filters.setdefault(key, {}).setdefault(prefix, set())
            for raw in block.tests:
                if _target_in_filter_subtree(_entry_target(raw), prefix):
                    ids.add(raw)
    return block_filters, unmatched


def resolve_affected_stages(
    block_filters: dict[tuple[str, int], dict[str, set[str]]],
    yaml_index: YAMLIndex,
    stages_by_yaml: dict[str, list[tuple[str, Stage]]],
) -> set[str]:
    """Map block_filters keys → Jenkins stage names via mako condition match."""
    block_by_key = {(b.yaml_stem, b.block_index): b for b in yaml_index.blocks}
    affected: set[str] = set()
    for key in block_filters:
        block = block_by_key.get(key)
        if block is None:
            continue
        for stage_name, stage in stages_by_yaml.get(block.yaml_stem, []):
            if block_matches_stage(block, stage):
                affected.add(stage_name)
    return affected
