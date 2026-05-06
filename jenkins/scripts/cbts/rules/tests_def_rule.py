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
"""TestsDefRule — narrows CI when .py test files under tests/ change.

YAMLIndex is the source of truth for what's narrowable; this rule does
not classify files by name. AST scope mapping yields function-level
anchors (`file::TestC::test_m`) when every changed line lands in a
pytest test scope; otherwise falls back to file-level. conftest.py and
__init__.py anchor on their enclosing directory.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Optional

from blocks import Stage, YAMLIndex

from ._helpers import (
    iter_diff_post_line_numbers,
    lookup_paths_into_block_filters,
    resolve_affected_stages,
    stages_by_yaml_stem,
)
from .base import PRInputs, Rule, RuleResult

# Changes touching at least this fraction of all blocks (top-level conftest,
# common.py, …) are too broad to narrow usefully and trigger fallback.
BLAST_RADIUS_FRACTION = 0.8


def _map_lines_to_pytest_scopes(content: str, line_numbers: set[int]) -> Optional[set[str]]:
    """Resolve each line to its enclosing pytest scope.

    Returns scope strings — `test_X` (top-level), `TestC::test_m` (class
    method), or `TestC` (class body, e.g. `setup_method`). Returns None
    if any line is module-level / outside any `Test*` class — caller
    falls back to file-level narrowing.
    """
    try:
        tree = ast.parse(content)
    except (SyntaxError, ValueError):
        return None

    line_to_scope: dict[int, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("test"):
                continue
            end = node.end_lineno or node.lineno
            for ln in range(node.lineno, end + 1):
                line_to_scope[ln] = node.name
        elif isinstance(node, ast.ClassDef):
            if not node.name.startswith("Test"):
                continue
            class_end = node.end_lineno or node.lineno
            for ln in range(node.lineno, class_end + 1):
                line_to_scope[ln] = node.name
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if not child.name.startswith("test"):
                        continue
                    method_end = child.end_lineno or child.lineno
                    method_scope = f"{node.name}::{child.name}"
                    for ln in range(child.lineno, method_end + 1):
                        line_to_scope[ln] = method_scope

    scopes: set[str] = set()
    for ln in line_numbers:
        scope = line_to_scope.get(ln)
        if scope is None:
            return None
        scopes.add(scope)
    return scopes if scopes else None


def _is_perf_stem(stem: str) -> bool:
    return stem == "l0_perf" or "perf_sanity" in stem


class TestsDefRule(Rule):
    name = "testdef"
    needs_diff_for: tuple[str, ...] = ("tests/**/*.py",)

    def __init__(
        self,
        yaml_index: YAMLIndex,
        stages: dict[str, Stage],
        repo_root: Path,
    ) -> None:
        self.yaml_index = yaml_index
        self._stages_by_yaml = stages_by_yaml_stem(stages)
        self._repo_root = repo_root
        self._total_blocks = len(yaml_index.blocks)

    def _compute_anchors(self, git_path: str, yaml_path: str, diff: str) -> list[str]:
        """Return lookup anchors for one file.

        File-level fallback when no diff, file unreadable, AST parse
        fails, or any line is module-level.
        """
        if not diff:
            return [yaml_path]
        try:
            content = (self._repo_root / git_path).read_text(encoding="utf-8")
        except OSError:
            return [yaml_path]
        line_numbers = iter_diff_post_line_numbers(diff)
        if not line_numbers:
            return [yaml_path]
        scopes = _map_lines_to_pytest_scopes(content, line_numbers)
        if scopes is None:
            return [yaml_path]
        return [f"{yaml_path}::{s}" for s in sorted(scopes)]

    def apply(self, pr: PRInputs) -> Optional[RuleResult]:
        candidates = [f for f in pr.changed_files if f.endswith(".py") and f.startswith("tests/")]
        if not candidates:
            return None

        block_filters: dict[tuple[str, int], dict[str, set[str]]] = {}
        narrowed: set[str] = set()
        handled: set[str] = set()
        out_of_namespace: set[str] = set()
        no_match: set[str] = set()

        for git_path in candidates:
            yaml_path = self.yaml_index.git_path_to_yaml_key(git_path)
            if yaml_path is None:
                # Outside any YAML namespace — could still impact selection
                # via implicit pytest discovery (top-level conftest, helper
                # modules outside YAML's view). Don't claim; let Selector
                # report it as Unhandled → fallback.
                out_of_namespace.add(git_path)
                continue
            handled.add(git_path)
            anchors = self._compute_anchors(git_path, yaml_path, pr.diffs.get(git_path, ""))
            file_bf, _ = lookup_paths_into_block_filters(self.yaml_index, anchors)
            if not file_bf:
                # In-namespace but no YAML entry covers this path → no
                # narrow contribution; claimed as noop.
                no_match.add(git_path)
                continue
            narrowed.add(git_path)
            for key, prefix_to_ids in file_bf.items():
                dst = block_filters.setdefault(key, {})
                for prefix, ids in prefix_to_ids.items():
                    dst.setdefault(prefix, set()).update(ids)

        if not handled:
            # All candidates were out-of-namespace → don't claim; fallback.
            return None

        if not block_filters:
            # Every claimed (in-namespace) path had no covering YAML entry:
            # the rule fired and decided no test-selection impact.
            return RuleResult(
                handled_files=handled,
                affected_stages=set(),
                scope="noop",
                sanity_relevant=False,
                perfsanity_relevant=False,
                reason=f"testdef: {len(handled)} path(s) → no covering YAML entry",
            )

        if (
            self._total_blocks > 0
            and len(block_filters) >= self._total_blocks * BLAST_RADIUS_FRACTION
        ):
            return RuleResult(
                handled_files=handled,
                affected_stages=set(),
                scope=None,
                reason=(
                    f"testdef: blast-radius cap "
                    f"({len(block_filters)}/{self._total_blocks} blocks "
                    f">= {BLAST_RADIUS_FRACTION:.0%}); fallback"
                ),
            )

        affected_stages = resolve_affected_stages(
            block_filters, self.yaml_index, self._stages_by_yaml
        )
        sanity_relevant = any(stem == "l0_sanity_check" for stem, _ in block_filters)
        perfsanity_relevant = any(_is_perf_stem(stem) for stem, _ in block_filters)

        nonarrow_note = ""
        if no_match:
            nonarrow_note = f"; {len(no_match)} in-namespace path(s) with no covering YAML entry"

        return RuleResult(
            handled_files=handled,
            affected_stages=affected_stages,
            scope="testdefonly",
            block_filters=block_filters,
            sanity_relevant=sanity_relevant,
            perfsanity_relevant=perfsanity_relevant,
            reason=(
                f"testdef: {len(narrowed)} path(s) → "
                f"{len(block_filters)} blocks, {len(affected_stages)} stages"
                f"{nonarrow_note}"
            ),
        )
