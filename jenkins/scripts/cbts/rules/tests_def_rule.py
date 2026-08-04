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
"""TestsDefRule — narrows CI when files under tests/ change.

Handles any path under `tests/` whose first component (after the
`tests/integration/defs/` strip) appears in some YAML entry's namespace.
For `test_*.py` files, AST scope mapping yields function-level anchors
(`file::TestC::test_m`) when every changed line lands in a pytest scope;
otherwise file-level. For non-test_*.py paths (conftest, helpers, data
files like `references/*.yaml` or `disaggregated/test_configs/*.yaml`),
`find_match_for_path` walks up enclosing directories to the narrowest
YAML-covered ancestor.

Paths matched by `out_of_scope_rule.is_out_of_scope` (QA / dev test
lists, `.test_durations`, `microbenchmarks/`, `tests/**/*.md`) are
excluded from candidates so `OutOfScopeRule`'s noop claim is not
overridden by a same-file narrow contribution.
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
from .out_of_scope_rule import is_out_of_scope

# Changes touching at least this fraction of all blocks (top-level conftest,
# common.py, …) are too broad to narrow usefully and trigger fallback.
BLAST_RADIUS_FRACTION = 0.8

ACCURACY_REFS_PREFIX = "tests/integration/defs/accuracy/references/"
ACCURACY_DIR = "tests/integration/defs/accuracy"


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


def _diff_has_deletions(diff: str) -> bool:
    """True if any `-` content line (not the `---` file header) appears."""
    for line in diff.splitlines():
        if not line or line.startswith(("+++", "---", "@@")):
            continue
        if line.startswith("-"):
            return True
    return False


def _yaml_top_keys_for_lines(content: str, line_numbers: set[int]) -> set[str]:
    """Return the set of top-level YAML keys whose section contains any line in `line_numbers`.

    A "top-level key" is an unindented line of the form `<key>:`
    (optional trailing comment).
    """
    keys: list[Optional[str]] = []
    current: Optional[str] = None
    for raw in content.splitlines():
        if raw and not raw[0].isspace() and not raw.lstrip().startswith("#"):
            stripped = raw.split("#", 1)[0].rstrip()
            if stripped.endswith(":"):
                current = stripped[:-1].strip().strip("'\"")
        keys.append(current)
    return {keys[i - 1] for i in line_numbers if 1 <= i <= len(keys) and keys[i - 1]}


class TestsDefRule(Rule):
    name = "testdef"
    needs_diff_for: tuple[str, ...] = ("tests/**/*",)

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

        File-level fallback when no diff, file unreadable or not UTF-8
        (e.g. binary fixtures), AST parse fails, or any line is module-
        level. accuracy/references/*.yaml diffs are refined to per-test-
        class anchors via the model-name mapping in
        `_compute_accuracy_reference_anchors`.
        """
        if git_path.startswith(ACCURACY_REFS_PREFIX) and git_path.endswith((".yaml", ".yml")):
            return self._compute_accuracy_reference_anchors(git_path, yaml_path, diff)
        if not diff:
            return [yaml_path]
        try:
            content = (self._repo_root / git_path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return [yaml_path]
        line_numbers = iter_diff_post_line_numbers(diff)
        if not line_numbers:
            return [yaml_path]
        scopes = _map_lines_to_pytest_scopes(content, line_numbers)
        if scopes is None:
            return [yaml_path]
        return [f"{yaml_path}::{s}" for s in sorted(scopes)]

    def _compute_accuracy_reference_anchors(
        self, git_path: str, yaml_path: str, diff: str
    ) -> list[str]:
        """Map a `references/<dataset>.yaml` diff to per-class anchors.

        Each top-level YAML key is a HF model name; map those to test
        classes via the `MODEL_NAME = "<hf>"` literal in
        `accuracy/test_*.py`. Class-level anchors let
        `find_match_for_path`'s lineage walk match every parametrization
        of those classes. Falls back to `[yaml_path]` (→ dir walk-up to
        `accuracy/`) when refinement isn't possible.

        Refinement is only sound when every changed line has a post-
        image position whose top-level key can be read directly. A `-`
        line has no post-image position; `iter_diff_post_line_numbers`
        anchors it to the next surviving line, which may belong to a
        different model section (e.g. deleting `ModelA:` and its body
        attributes those `-` lines to the start of `ModelB:`). Any
        deletion therefore triggers fallback.
        """
        if not diff:
            return [yaml_path]
        if _diff_has_deletions(diff):
            return [yaml_path]
        line_numbers = iter_diff_post_line_numbers(diff)
        if not line_numbers:
            return [yaml_path]
        try:
            content = (self._repo_root / git_path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return [yaml_path]
        changed_models = _yaml_top_keys_for_lines(content, line_numbers)
        if not changed_models:
            return [yaml_path]
        model_map = self._accuracy_model_to_classes()
        anchors = sorted({a for m in changed_models for a in model_map.get(m, ())})
        return anchors or [yaml_path]

    def _accuracy_model_to_classes(self) -> dict[str, list[str]]:
        """Cached: HF model name → list of `accuracy/test_X.py::ClassName`.

        Built by AST-scanning `accuracy/test_*.py` for `class TestX:` with
        a literal `MODEL_NAME = "<hf>"` assignment.
        """
        cached = getattr(self, "_acc_model_map", None)
        if cached is not None:
            return cached
        out: dict[str, list[str]] = {}
        acc_dir = self._repo_root / ACCURACY_DIR
        if acc_dir.is_dir():
            for py in sorted(acc_dir.glob("test_*.py")):
                try:
                    tree = ast.parse(py.read_text(encoding="utf-8"))
                except (OSError, SyntaxError, UnicodeDecodeError):
                    continue
                rel = f"accuracy/{py.name}"
                for node in tree.body:
                    if not isinstance(node, ast.ClassDef) or not node.name.startswith("Test"):
                        continue
                    for child in node.body:
                        if not isinstance(child, ast.Assign):
                            continue
                        if not any(
                            isinstance(t, ast.Name) and t.id == "MODEL_NAME" for t in child.targets
                        ):
                            continue
                        v = child.value
                        if isinstance(v, ast.Constant) and isinstance(v.value, str):
                            out.setdefault(v.value, []).append(f"{rel}::{node.name}")
                            break
        self._acc_model_map = out
        return out

    def apply(self, pr: PRInputs) -> Optional[RuleResult]:
        candidates = [
            f for f in pr.changed_files if f.startswith("tests/") and not is_out_of_scope(f)
        ]
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
                base = git_path.rsplit("/", 1)[-1]
                if base.startswith("test_") and base.endswith(".py"):
                    # Standalone test_*.py not referenced by any L0 YAML —
                    # pytest doesn't auto-import test files into other
                    # tests, so this file's edits can't affect what L0
                    # runs. Claim as noop contribution.
                    handled.add(git_path)
                    no_match.add(git_path)
                    continue
                # conftest / __init__ / helper / data — could impact
                # selection via implicit pytest discovery; let Selector
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
            nonarrow_note = f"; {len(no_match)} path(s) not in any L0 YAML"

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
