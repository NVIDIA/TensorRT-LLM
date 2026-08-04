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
_COMMENT_BODY_RE = re.compile(r"^\s*#")


def strip_noop_diff_lines(diff: str) -> str:
    """Normalize a unified diff so blank- and comment-only edits don't count.

    Blank or comment-only `+/-` lines carry no test-selection meaning for
    CBTS — they can't introduce imports, decorators, test methods, or
    YAML entries. Treating them as real changes causes anchor walk-up
    and stage-select detection to over-fire (e.g. a trailing `+` blank
    line in a method-only diff drops AST scope to module-level and
    forces a file-level anchor fallback).

    Normalization rules:
      - `+` blank / comment-only line → downgrade to context (` `).
        Cursor in `iter_diff_post_line_numbers` still advances, so
        every subsequent line keeps its correct post-image line number.
      - `-` blank / comment-only line → drop entirely. `-` lines never
        advance the post-image cursor, so removal is safe.
      - All real-content `+/-` lines, context lines, hunk headers, and
        file headers are emitted verbatim.

    Hunk header counts (`-A,B +C,D`) are left as-is — CBTS walkers
    don't consume B or D; only `+C` (the post-image start line) is
    read, and that remains correct because we preserve cursor steps
    via the context-downgrade trick.
    """
    out: list[str] = []
    for raw in diff.splitlines(keepends=True):
        body = raw.rstrip("\n").rstrip("\r")
        if not body or body.startswith(("+++", "---", "@@")):
            out.append(raw)
            continue
        sign = body[0]
        rest = body[1:]
        if sign in ("+", "-") and (not rest.strip() or _COMMENT_BODY_RE.match(rest)):
            if sign == "+":
                # downgrade to context: preserves post-image cursor
                out.append(" " + raw[1:])
            # else: drop entirely (no cursor impact)
            continue
        out.append(raw)
    return "".join(out)


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
